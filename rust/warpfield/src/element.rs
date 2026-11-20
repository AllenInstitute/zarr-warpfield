//! The two supported data types.

pub trait Element: Copy + Send + Sync + 'static {
    /// zarr / numpy name of the type.
    const NAME: &'static str;
    const SIZE: usize;
    fn to_f64(self) -> f64;
    /// Round a float64 to this type (round to nearest even, overflow to ±inf).
    fn from_f64(v: f64) -> Self;
    /// `v` rounded to this type and widened back to float64.
    #[inline(always)]
    fn round(v: f64) -> f64 {
        Self::from_f64(v).to_f64()
    }
    fn write_le(self, out: &mut Vec<u8>);
    fn read_le(b: &[u8]) -> Self;
}

impl Element for f32 {
    const NAME: &'static str = "float32";
    const SIZE: usize = 4;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f32 {
        v as f32
    }
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes())
    }
    fn read_le(b: &[u8]) -> f32 {
        f32::from_le_bytes(b.try_into().unwrap())
    }
}

impl Element for f64 {
    const NAME: &'static str = "float64";
    const SIZE: usize = 8;
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn round(v: f64) -> f64 {
        v
    }
    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes())
    }
    fn read_le(b: &[u8]) -> f64 {
        f64::from_le_bytes(b.try_into().unwrap())
    }
}
