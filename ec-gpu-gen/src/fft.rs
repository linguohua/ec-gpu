use std::cmp;
use std::sync::{Arc, Mutex, RwLock};

use ec_gpu::GpuName;
use ff::Field;
use log::{error, info};
use rust_gpu_tools::{program_closures, Device, LocalBuffer, Program};

use crate::error::{EcError, EcResult};
use crate::threadpool::{GPU_PROGRAM_LOCKS, NUM_THREADS, THREAD_POOL};

const LOG2_MAX_ELEMENTS: usize = 32; // At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 8; // Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 7; // 128

enum ECMemSizeType {
    Huge,
    Normal,
    Small,
}

/// FFT kernel for a single GPU.
pub struct SingleFftKernel<'a, F>
where
    F: Field + GpuName,
{
    mem_size_type: ECMemSizeType,
    device_id: String,
    gpu_lock: Arc<Mutex<()>>,
    program: Program,
    /// An optional function which will be called at places where it is possible to abort the FFT
    /// calculations. If it returns true, the calculation will be aborted with an
    /// [`EcError::Aborted`].
    maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    _phantom: std::marker::PhantomData<F>,
}

impl<'a, F: Field + GpuName> SingleFftKernel<'a, F> {
    /// Create a new FFT instance for the given device.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create(
        program: Program,
        device: &Device,
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let gpu_id = device.unique_id().to_string();
        let lck = {
            let mut llock = GPU_PROGRAM_LOCKS.lock().unwrap();
            let lck = llock
                .entry(gpu_id.to_string())
                .or_insert(Arc::new(Mutex::new(())));
            lck.clone()
        };

        let mem_size = device.memory();
        let mem_type = if mem_size >= (20 * 1024 * 1024 * 1024) {
            ECMemSizeType::Huge
        } else if mem_size >= (10 * 1024 * 1024 * 1024) {
            ECMemSizeType::Normal
        } else {
            ECMemSizeType::Small
        };

        Ok(SingleFftKernel {
            program: program,
            gpu_lock: lck,
            device_id: gpu_id,
            mem_size_type: mem_type,
            maybe_abort,
            _phantom: Default::default(),
        })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        match self.mem_size_type {
            ECMemSizeType::Huge => self.radix_fft1(input, omega, log_n),
            ECMemSizeType::Normal => self.radix_fft2(input, omega, log_n),
            ECMemSizeType::Small => self.radix_fft3(input, omega, log_n),
        }
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft1(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        let closures = program_closures!(|program, input: &mut [F]| -> EcResult<()> {
            let n = 1 << log_n;

            let lock = self.gpu_lock.clone();
            let lock2 = lock.lock().unwrap();

            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![F::zero(); 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = F::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![F::zero(); 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<F>(n)? };
            program.write_from_buffer(&mut src_buffer, &*input)?;

            // let lock = self.gpu_lock.clone();
            // let lock2 = lock.lock().unwrap();

            let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };

            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                if let Some(maybe_abort) = &self.maybe_abort {
                    if maybe_abort() {
                        return Err(EcError::Aborted);
                    }
                }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel_name = format!("{}_radix_fft", F::name());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<F>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            drop(dst_buffer);
            drop(lock2);

            program.read_into_buffer(&src_buffer, input)?;

            Ok(())
        });

        self.program.run(closures, input)
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft2(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        let closures = program_closures!(|program, input: &mut [F]| -> EcResult<()> {
            let n = 1 << log_n;
            let lock = self.gpu_lock.clone();
            let lock2 = lock.lock().unwrap();

            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![F::zero(); 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = F::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![F::zero(); 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            // let lock = self.gpu_lock.clone();
            // let lock2 = lock.lock().unwrap();

            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<F>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };

            program.write_from_buffer(&mut src_buffer, &*input)?;
            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                if let Some(maybe_abort) = &self.maybe_abort {
                    if maybe_abort() {
                        return Err(EcError::Aborted);
                    }
                }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel_name = format!("{}_radix_fft", F::name());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<F>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input)?;

            drop(dst_buffer);
            drop(src_buffer);
            drop(lock2);

            Ok(())
        });

        self.program.run(closures, input)
    }

    fn radix_fft3(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        let lock = self.gpu_lock.clone();
        let _lock2 = lock.lock().unwrap();

        let n = input.len();

        let mut evens = Vec::with_capacity(n / 2);
        let mut odds = Vec::with_capacity(n / 2);

        let chunk_count = cmp::max(*NUM_THREADS - 4, 1);
        let chunk_size = (n / 2) / chunk_count;

        // even and odd to half array
        unsafe {
            let evens_slice = std::slice::from_raw_parts_mut(evens.as_mut_ptr(), n / 2);
            let odds_slice = std::slice::from_raw_parts_mut(odds.as_mut_ptr(), n / 2);

            THREAD_POOL.scoped(|s| {
                for ((es, os), ip) in evens_slice
                    .chunks_mut(chunk_size)
                    .zip(odds_slice.chunks_mut(chunk_size))
                    .zip(input.chunks(chunk_size * 2))
                {
                    s.execute(move || {
                        for i in 0..es.len() {
                            es[i] = ip[i * 2];
                            os[i] = ip[i * 2 + 1];
                        }
                    });
                }
            });

            evens.set_len(n / 2);
            odds.set_len(n / 2);
        }

        // call gpu to do double halfs fft
        THREAD_POOL.scoped(|s| -> EcResult<()> {
            for (index, ip) in input[0..n / 2].chunks_mut(chunk_size).enumerate() {
                s.execute(move || {
                    let mut w_m = if index > 0 {
                        omega.pow_vartime(&[(index * chunk_size) as u64])
                    } else {
                        F::one()
                    };

                    for i in 0..ip.len() {
                        ip[i] = w_m;
                        w_m = w_m * omega;
                    }
                });
            }

            let omega_double = omega.square();
            self.radix_fft_official(&mut evens[..], &omega_double, log_n - 1)?;
            self.radix_fft_official(&mut odds[..], &omega_double, log_n - 1)?;
            Ok(())
        })?;

        // odds
        THREAD_POOL.scoped(|s| {
            for (os, ip) in odds
                .chunks_mut(chunk_size)
                .zip(input[0..n / 2].chunks(chunk_size))
            {
                s.execute(move || {
                    for i in 0..os.len() {
                        os[i] = os[i] * ip[i];
                    }
                });
            }
        });

        // low half output
        THREAD_POOL.scoped(|s| {
            for ((es, os), ip) in evens
                .chunks(chunk_size)
                .zip(odds.chunks(chunk_size))
                .zip(input[..n / 2].chunks_mut(chunk_size))
            {
                s.execute(move || {
                    for i in 0..es.len() {
                        ip[i] = es[i] + os[i]
                    }
                });
            }
        });

        // high half output
        THREAD_POOL.scoped(|s| {
            for ((es, os), ip) in evens
                .chunks(chunk_size)
                .zip(odds.chunks(chunk_size))
                .zip(input[n / 2..].chunks_mut(chunk_size))
            {
                s.execute(move || {
                    for i in 0..es.len() {
                        ip[i] = es[i] - os[i]
                    }
                });
            }
        });

        Ok(())
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft_official(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        let closures = program_closures!(|program, input: &mut [F]| -> EcResult<()> {
            let n = 1 << log_n;
            // All usages are safe as the buffers are initialized from either the host or the GPU
            // before they are read.
            let mut src_buffer = unsafe { program.create_buffer::<F>(n)? };
            let mut dst_buffer = unsafe { program.create_buffer::<F>(n)? };
            // The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);

            // Precalculate:
            // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
            let mut pq = vec![F::zero(); 1 << max_deg >> 1];
            let twiddle = omega.pow_vartime([(n >> max_deg) as u64]);
            pq[0] = F::one();
            if max_deg > 1 {
                pq[1] = twiddle;
                for i in 2..(1 << max_deg >> 1) {
                    pq[i] = pq[i - 1];
                    pq[i].mul_assign(&twiddle);
                }
            }
            let pq_buffer = program.create_buffer_from_slice(&pq)?;

            // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
            let mut omegas = vec![F::zero(); 32];
            omegas[0] = *omega;
            for i in 1..LOG2_MAX_ELEMENTS {
                omegas[i] = omegas[i - 1].pow_vartime([2u64]);
            }
            let omegas_buffer = program.create_buffer_from_slice(&omegas)?;

            program.write_from_buffer(&mut src_buffer, &*input)?;
            // Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
            let mut log_p = 0u32;
            // Each iteration performs a FFT round
            while log_p < log_n {
                if let Some(maybe_abort) = &self.maybe_abort {
                    if maybe_abort() {
                        return Err(EcError::Aborted);
                    }
                }

                // 1=>radix2, 2=>radix4, 3=>radix8, ...
                let deg = cmp::min(max_deg, log_n - log_p);

                let n = 1u32 << log_n;
                let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
                let global_work_size = n >> deg;
                let kernel_name = format!("{}_radix_fft", F::name());
                let kernel = program.create_kernel(
                    &kernel_name,
                    global_work_size as usize,
                    local_work_size as usize,
                )?;
                kernel
                    .arg(&src_buffer)
                    .arg(&dst_buffer)
                    .arg(&pq_buffer)
                    .arg(&omegas_buffer)
                    .arg(&LocalBuffer::<F>::new(1 << deg))
                    .arg(&n)
                    .arg(&log_p)
                    .arg(&deg)
                    .arg(&max_deg)
                    .run()?;

                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            program.read_into_buffer(&src_buffer, input)?;

            Ok(())
        });

        self.program.run(closures, input)
    }
}

/// One FFT kernel for each GPU available.
pub struct FftKernel<'a, F>
where
    F: Field + GpuName,
{
    kernels: Vec<SingleFftKernel<'a, F>>,
}

impl<'a, F> FftKernel<'a, F>
where
    F: Field + GpuName,
{
    /// Create new kernels, one for each given device.
    pub fn create(programs: Vec<Program>, devices: &[&Device]) -> EcResult<Self> {
        Self::create_optional_abort(programs, devices, None)
    }

    /// Create new kernels, one for each given device, with early abort hook.
    ///
    /// The `maybe_abort` function is called when it is possible to abort the computation, without
    /// leaving the GPU in a weird state. If that function returns `true`, execution is aborted.
    pub fn create_with_abort(
        programs: Vec<Program>,
        devices: &[&Device],
        maybe_abort: &'a (dyn Fn() -> bool + Send + Sync),
    ) -> EcResult<Self> {
        Self::create_optional_abort(programs, devices, Some(maybe_abort))
    }

    fn create_optional_abort(
        programs: Vec<Program>,
        devices: &[&Device],
        maybe_abort: Option<&'a (dyn Fn() -> bool + Send + Sync)>,
    ) -> EcResult<Self> {
        let kernels: Vec<_> = programs
            .into_iter()
            .zip(devices.iter())
            .filter_map(|(program, device)| {
                let device_name = program.device_name().to_string();
                let device_id = device.unique_id().to_string();
                let kernel = SingleFftKernel::<F>::create(program, device, maybe_abort);
                if let Err(ref e) = kernel {
                    error!(
                        "Cannot initialize kernel for device '{}' id '{}'! Error: {}",
                        device_name, device_id, e
                    );
                }
                kernel.ok()
            })
            .collect();

        if kernels.is_empty() {
            return Err(EcError::Simple("No working GPUs found!"));
        }
        info!("FFT: {} working device(s) selected. ", kernels.len());
        for (i, k) in kernels.iter().enumerate() {
            info!(
                "FFT: Device {}: {} id: {}",
                i,
                k.program.device_name(),
                k.device_id
            );
        }

        Ok(Self { kernels })
    }

    /// Performs FFT on `input`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses the first available GPU.
    pub fn radix_fft(&mut self, input: &mut [F], omega: &F, log_n: u32) -> EcResult<()> {
        self.kernels[0].radix_fft(input, omega, log_n)
    }

    /// Performs FFT on `inputs`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    ///
    /// Uses all available GPUs to distribute the work.
    pub fn radix_fft_many(
        &mut self,
        inputs: &mut [&mut [F]],
        omegas: &[F],
        log_ns: &[u32],
    ) -> EcResult<()> {
        let n = inputs.len();
        let num_devices = self.kernels.len();
        let chunk_size = ((n as f64) / (num_devices as f64)).ceil() as usize;

        let result = Arc::new(RwLock::new(Ok(())));

        THREAD_POOL.scoped(|s| {
            for (((inputs, omegas), log_ns), kern) in inputs
                .chunks_mut(chunk_size)
                .zip(omegas.chunks(chunk_size))
                .zip(log_ns.chunks(chunk_size))
                .zip(self.kernels.iter_mut())
            {
                let result = result.clone();
                s.execute(move || {
                    for ((input, omega), log_n) in
                        inputs.iter_mut().zip(omegas.iter()).zip(log_ns.iter())
                    {
                        if result.read().unwrap().is_err() {
                            break;
                        }

                        if let Err(err) = kern.radix_fft(input, omega, *log_n) {
                            *result.write().unwrap() = Err(err);
                            break;
                        }
                    }
                });
            }
        });

        Arc::try_unwrap(result).unwrap().into_inner().unwrap()
    }
}
