use std::fmt;

/// Errors from configuration checks (`Spec`) and from malformed chunks (`Format`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The configuration or the chunk shape/dtype is not allowed by the spec.
    Spec(String),
    /// The encoded bytes are not a valid warpfield chunk for this shape/config.
    Format(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Spec(m) => write!(f, "warpfield spec error: {m}"),
            Error::Format(m) => write!(f, "warpfield format error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn spec<T>(m: impl Into<String>) -> Result<T> {
    Err(Error::Spec(m.into()))
}

pub(crate) fn format<T>(m: impl Into<String>) -> Result<T> {
    Err(Error::Format(m.into()))
}
