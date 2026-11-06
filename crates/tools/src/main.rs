//! `gyroship-tools`: developer command line for Gyroship.

use clap::Parser;
use std::process::ExitCode;

#[derive(Parser)]
#[command(name = "gyroship-tools", version, about = "Gyroship developer tools")]
struct Cli {}

fn main() -> ExitCode {
    Cli::parse();
    ExitCode::SUCCESS
}
