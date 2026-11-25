//! Makes one host compute pipeline, alone in its process (a host compiler
//! that crashes takes only this process down):
//!
//! ```text
//! cargo run --release -p victory-shader --features host --example one_pipeline -- <file> [out.dxil]
//! ```
//!
//! `<file>` is either a title's shader container (`<hash>.bin`, as
//! `translate --cache` writes them with `VICTORY_CONTAINER_DUMP`), translated
//! here with the root signature the tests use, its DXIL written to
//! `out.dxil` if given; or a dumped `<hash>.dxil` with its root signature
//! `<hash>.rts` next to it (`VICTORY_HOST_DXIL_DUMP`, compute pipelines).
//! Prints `ok` or the error; exit code 0 when the pipeline was made.

use victory_shader::emit::host::Host;
use windows::Win32::Graphics::Direct3D12::ID3D12RootSignature;

fn run(path: &std::path::Path, out: Option<&std::path::Path>) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let host = Host::new(false)?;
    if path.extension().is_some_and(|e| e == "dxil") {
        let rts_path = path.with_extension("rts");
        let rts = std::fs::read(&rts_path).map_err(|e| format!("{}: {e}", rts_path.display()))?;
        // SAFETY: a serialized root signature read in full.
        let root: ID3D12RootSignature = unsafe { host.device.CreateRootSignature(0, &rts) }
            .map_err(|e| format!("root signature: {}", e.message()))?;
        return host.compute_pipeline_with(&root, &bytes).map(drop);
    }
    let shader = victory_shader::emit::translate(&bytes).map_err(|e| format!("translate: {e}"))?;
    if let Some(out) = out {
        std::fs::write(out, &shader.dxil).map_err(|e| format!("{}: {e}", out.display()))?;
    }
    if std::env::var_os("ONE_PIPELINE_VALIDATE").is_some() {
        use victory_shader::emit::validator::Validator;
        let mut v = Validator::load(Validator::default_path().as_deref())?;
        let mut dxil = shader.dxil.clone();
        match v.validate(&mut dxil) {
            Ok(()) => eprintln!("validator: ok"),
            Err(e) => eprintln!("validator: {e}"),
        }
    }
    if std::env::var_os("ONE_PIPELINE_NO_HOST").is_some() {
        return Ok(());
    }
    host.compute_pipeline(&shader).map(drop)
}

/// `module+offset` of a code address.
fn place(pc: u64) -> String {
    use windows_sys::Win32::System::LibraryLoader::*;
    let mut module = std::ptr::null_mut();
    // SAFETY: plain lookups with valid out pointers.
    unsafe {
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            pc as usize as *const u16,
            &mut module,
        ) == 0
        {
            return format!("{pc:#x}");
        }
        let mut name = [0u16; 260];
        let n = GetModuleFileNameW(module, name.as_mut_ptr(), 260) as usize;
        let name = String::from_utf16_lossy(&name[..n]);
        let short = name.rsplit('\\').next().unwrap_or(&name).to_owned();
        format!("{short}+{:#x}", pc - module as u64)
    }
}

/// On an access violation, prints the faulting address and the unwound
/// stack (module+offset per frame) to stderr.
unsafe extern "system" fn on_fault(
    info: *mut windows_sys::Win32::System::Diagnostics::Debug::EXCEPTION_POINTERS,
) -> i32 {
    use windows_sys::Win32::System::Diagnostics::Debug::*;
    // SAFETY: the system passes a live record and context.
    unsafe {
        let record = &*(*info).ExceptionRecord;
        if record.ExceptionCode != windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION {
            return 0; // EXCEPTION_CONTINUE_SEARCH
        }
        let mut ctx = *(*info).ContextRecord;
        eprintln!(
            "fault: {} reading {:#x}; rax {:#x} rbx {:#x} rcx {:#x} rdx {:#x} rsi {:#x} rdi {:#x}",
            place(ctx.Rip),
            record.ExceptionInformation[1],
            ctx.Rax,
            ctx.Rbx,
            ctx.Rcx,
            ctx.Rdx,
            ctx.Rsi,
            ctx.Rdi
        );
        // At amdxc64.dll+0x861b9f, r14 is the instruction visited and rdx
        // the operand whose definition (at +0x10) has no record yet.
        let peek = |a: u64| -> Option<u64> {
            let mut v = 0u64;
            let mut n = 0usize;
            (ReadProcessMemory(
                -1isize as _,
                a as usize as *const _,
                (&raw mut v).cast(),
                8,
                &mut n,
            ) != 0)
                .then_some(v)
        };
        let describe = |what: &str, i: u64| {
            let op = peek(i + 0x84).map(|v| v as u32);
            let flags = peek(i + 0x88).map(|v| v as u32);
            let n = peek(i + 0x28).map(|v| v as u32);
            let block = peek(i + 0x70).and_then(|b| peek(b + 0xec)).map(|v| v as u32);
            eprintln!(
                "  {what} {i:#x}: opcode {op:x?} flags {flags:x?} operands {n:?} block {block:?}"
            );
        };
        describe("instruction", ctx.R14);
        if let Some(def) = peek(ctx.Rdx + 0x10) {
            let kind = peek(ctx.Rdx).map(|v| v as u32);
            eprintln!("  operand {:#x}: kind {kind:x?}", ctx.Rdx);
            describe("definition", def);
            // The instruction's operands, kind and defining opcode.
            if let (Some(list), Some(n)) = (peek(ctx.R14 + 0x20), peek(ctx.R14 + 0x28)) {
                for k in 0..(n as u32).min(16) as u64 {
                    let Some(o) = peek(list + 16 * k) else { break };
                    let kind = peek(o).map(|v| v as u32);
                    let d = peek(o + 0x10).unwrap_or(0);
                    let dop = peek(d + 0x84).map(|v| v as u32);
                    let db = peek(d + 0x70).and_then(|b| peek(b + 0xec)).map(|v| v as u32);
                    let rec = peek(d + 0x78).and_then(|r| peek(r + 0x10));
                    eprintln!(
                        "    operand {k}: kind {kind:x?} def {d:#x} opcode {dop:x?} block {db:?} record {rec:x?}"
                    );
                }
            }
        }
        // The block's instructions in order (first at block+0x28, next at
        // +0x10): opcode, operand count, and each operand's defining opcode.
        if let Some(block) = peek(ctx.R14 + 0x70) {
            let mut i = peek(block + 0x28).unwrap_or(0);
            let mut line = String::new();
            for _ in 0..400 {
                if i == 0 || peek(i + 0x10).unwrap_or(0) == 0 {
                    break;
                }
                let op = peek(i + 0x84).unwrap_or(0) as u32;
                let n = peek(i + 0x28).unwrap_or(0) as u32;
                let mark = if i == ctx.R14 {
                    "<USER>"
                } else if i == ctx.Rdx {
                    "<DEF>"
                } else {
                    ""
                };
                let mut ops = Vec::new();
                if let Some(list) = peek(i + 0x20) {
                    for k in 0..u64::from(n.min(6)) {
                        let o = peek(list + 16 * k).unwrap_or(0);
                        let d = peek(o + 0x10).unwrap_or(0);
                        let tag = if d == ctx.Rdx { "*" } else { "" };
                        ops.push(format!("{tag}{:x}", peek(d + 0x84).unwrap_or(0) as u32));
                    }
                }
                line.push_str(&format!(" {mark}{op:x}({})", ops.join(",")));
                i = peek(i + 0x10).unwrap_or(0);
            }
            eprintln!("  block:{line}");
        }
        for _ in 0..200 {
            eprintln!("  at {}", place(ctx.Rip));
            let mut base = 0u64;
            let entry = RtlLookupFunctionEntry(ctx.Rip, &mut base, std::ptr::null_mut());
            if entry.is_null() {
                // A leaf: the return address is on top of the stack.
                ctx.Rip = *(ctx.Rsp as *const u64);
                ctx.Rsp += 8;
            } else {
                let mut data = std::ptr::null_mut();
                let mut frame = 0u64;
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    base,
                    ctx.Rip,
                    entry,
                    &mut ctx,
                    &mut data,
                    &mut frame,
                    std::ptr::null_mut(),
                );
            }
            if ctx.Rip == 0 {
                break;
            }
        }
        0
    }
}

fn main() {
    // SAFETY: registers a handler that lives as long as the process.
    unsafe {
        windows_sys::Win32::System::Diagnostics::Debug::AddVectoredExceptionHandler(1, Some(on_fault));
    }
    let mut args = std::env::args_os().skip(1).map(std::path::PathBuf::from);
    let Some(path) = args.next() else {
        eprintln!("usage: one_pipeline <hash>.bin|<hash>.dxil [out.dxil]");
        std::process::exit(2);
    };
    let out = args.next();
    match run(&path, out.as_deref()) {
        Ok(()) => println!("ok"),
        Err(e) => {
            println!("error: {e}");
            std::process::exit(1);
        }
    }
}
