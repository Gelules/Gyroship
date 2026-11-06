//! Runs a command and reports its wall time and the peak memory of the whole
//! process tree it starts (cargo, every rustc, the linker), from a Windows
//! job object that this process and so all its children belong to.

use std::process::Command;
use std::time::Instant;

pub struct Usage {
    pub seconds: f64,
    /// Peak commit of all processes of the job at once.
    pub peak_total: u64,
    /// Peak commit of the largest single process.
    pub peak_process: u64,
    pub success: bool,
}

#[cfg(windows)]
pub fn run(program: &str, args: &[String]) -> super::Result<Usage> {
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        QueryInformationJobObject,
    };
    use windows::Win32::System::Threading::GetCurrentProcess;
    use windows::core::PCWSTR;

    // SAFETY: plain Win32 calls on handles this function owns.
    let job = unsafe { CreateJobObjectW(None, PCWSTR::null()) }?;
    unsafe { AssignProcessToJobObject(job, GetCurrentProcess()) }?;
    let start = Instant::now();
    let status = Command::new(program).args(args).status()?;
    let seconds = start.elapsed().as_secs_f64();
    let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
    unsafe {
        QueryInformationJobObject(
            Some(job),
            JobObjectExtendedLimitInformation,
            &mut info as *mut _ as *mut std::ffi::c_void,
            std::mem::size_of_val(&info) as u32,
            None,
        )
    }?;
    Ok(Usage { seconds, peak_total: info.PeakJobMemoryUsed as u64, peak_process: info.PeakProcessMemoryUsed as u64, success: status.success() })
}

#[cfg(not(windows))]
pub fn run(_program: &str, _args: &[String]) -> super::Result<Usage> {
    Err("measure needs Windows job objects".into())
}
