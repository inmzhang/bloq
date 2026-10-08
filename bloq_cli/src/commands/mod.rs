use std::io::Write as _;
use std::path::Path;

use bloq_ir::Bloq;
use color_eyre::eyre::{self, WrapErr};

use crate::{BuiltInBackend, terminal};

pub(crate) mod compile;
pub(crate) mod completion;
pub(crate) mod emit;
pub(crate) mod gallery;
pub(crate) mod stats;
pub(crate) mod validate;
pub(crate) mod view;

/// Read a saved Bloq IR program, picking the codec by extension and falling
/// back to "try text, then binary" for a renamed artifact.
///
/// Neither decoder validates what it decodes (see `Bloq::from_binary`), so
/// callers acting on the result decide whether to run `Bloq::validate` first.
pub(crate) fn load_bloq_ir(input: &Path) -> eyre::Result<Bloq> {
    let bytes = std::fs::read(input).wrap_err_with(|| format!("read {}", input.display()))?;
    match input.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext == bloq_ir::BLOQ_BINARY_EXTENSION => Bloq::from_binary(&bytes)
            .wrap_err_with(|| format!("decode {} as binary Bloq IR", input.display())),
        Some(ext) if ext == bloq_ir::BLOQ_TEXT_EXTENSION => {
            let text = String::from_utf8(bytes)
                .wrap_err_with(|| format!("read {} as UTF-8 text", input.display()))?;
            Bloq::from_text(&text)
                .wrap_err_with(|| format!("parse {} as Bloq IR text", input.display()))
        }
        _ => match std::str::from_utf8(&bytes) {
            Ok(text) => Bloq::from_text(text).or_else(|_| decode_binary_fallback(input, &bytes)),
            Err(_) => decode_binary_fallback(input, &bytes),
        },
    }
}

fn decode_binary_fallback(input: &Path, bytes: &[u8]) -> eyre::Result<Bloq> {
    Bloq::from_binary(bytes).wrap_err_with(|| {
        format!(
            "decode {} as Bloq IR (tried text and binary; name it .{} or .{} to pick a codec)",
            input.display(),
            bloq_ir::BLOQ_TEXT_EXTENSION,
            bloq_ir::BLOQ_BINARY_EXTENSION
        )
    })
}

/// Reject an existing output path that names the input file through another
/// spelling, symbolic link, or hard link.
pub(crate) fn reject_input_output_alias(input: &Path, output: &Path) -> eyre::Result<()> {
    if output
        .try_exists()
        .wrap_err_with(|| format!("inspect output file {}", output.display()))?
        && same_file::is_same_file(input, output).wrap_err_with(|| {
            format!(
                "compare input file {} with output file {}",
                input.display(),
                output.display()
            )
        })?
    {
        eyre::bail!(
            "output file {} aliases the input file; choose a different output path",
            output.display()
        );
    }
    Ok(())
}

/// Text or binary output shared by compile and emit commands.
#[derive(Debug)]
enum EmittedArtifact {
    Text(String),
    Binary(Vec<u8>),
}

impl EmittedArtifact {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Text(text) => text.as_bytes(),
            Self::Binary(bytes) => bytes,
        }
    }

    fn write_stdout(&self) -> eyre::Result<()> {
        terminal::write_stdout(self.bytes()).wrap_err("write artifact to stdout")?;
        if let Self::Text(text) = self
            && !text.ends_with('\n')
        {
            terminal::write_stdout(b"\n").wrap_err("terminate text artifact on stdout")?;
        }
        Ok(())
    }
}

impl BuiltInBackend {
    fn emit(
        self,
        program: &Bloq,
        options: &bloq_stim::BloqStimOptions,
    ) -> eyre::Result<EmittedArtifact> {
        Ok(match self {
            Self::Stim => EmittedArtifact::Text(
                bloq_stim::emit_bloq_stim_with(program, options).wrap_err("emit Stim text")?,
            ),
            Self::IrText => EmittedArtifact::Text(program.to_text()),
            Self::IrBinary => EmittedArtifact::Binary(program.to_binary()),
        })
    }
}

/// Replace `path` only after a complete same-directory temporary file exists.
fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    atomic_write_with(path, |temporary| std::fs::write(temporary, bytes))
}

fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&Path) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "output path has no file name",
        )
    })?;
    let permissions = match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "refusing to replace symbolic-link output {}",
                    path.display()
                ),
            ));
        }
        Ok(metadata) => {
            // Replacing via rename only checks the parent directory. Preserve
            // the previous write contract by checking the target itself too.
            drop(std::fs::OpenOptions::new().write(true).open(path)?);
            Some(metadata.permissions())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(error),
    };

    let mut builder = tempfile::Builder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        // Match ordinary file creation: 0666 filtered through the process umask.
        builder.permissions(std::fs::Permissions::from_mode(0o666));
    }
    let mut temporary = builder.tempfile_in(parent)?;
    write(temporary.path())?;
    temporary.flush()?;
    if let Some(permissions) = permissions {
        temporary.as_file().set_permissions(permissions)?;
    }
    temporary
        .persist(path)
        .map(|_| ())
        .map_err(|error| error.error)
}
