//! Codec configuration (the `configuration` object of the zarr codec metadata).

use crate::error::{Result, spec};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Predicted across and included in the Jacobian.
    Spatial,
    /// Predicted across, excluded from the Jacobian (e.g. time, section index).
    Smooth,
    /// Not predicted across: each slice is coded on its own.
    Independent,
    /// The vector component axis.
    Component,
}

impl Role {
    pub fn parse(s: &str) -> Result<Role> {
        Ok(match s {
            "spatial" => Role::Spatial,
            "smooth" => Role::Smooth,
            "independent" => Role::Independent,
            "component" => Role::Component,
            _ => return spec(format!("unknown axis role {s:?}")),
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Role::Spatial => "spatial",
            Role::Smooth => "smooth",
            Role::Independent => "independent",
            Role::Component => "component",
        }
    }

    pub fn predicted(self) -> bool {
        matches!(self, Role::Spatial | Role::Smooth)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predictor {
    Linear,
    Cubic,
    Eno,
}

impl Predictor {
    pub fn parse(s: &str) -> Result<Predictor> {
        Ok(match s {
            "linear" => Predictor::Linear,
            "cubic" => Predictor::Cubic,
            "eno" => Predictor::Eno,
            _ => return spec(format!("predictor must be linear, cubic or eno, not {s:?}")),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Known set {0, s, 2s, ...} ∪ {N-1}: nothing is extrapolated (default).
    Clamped,
    /// Known set {0, s, 2s, ...}: SZ3-like baseline, the tail is extrapolated.
    Extrapolate,
}

impl Boundary {
    pub fn parse(s: &str) -> Result<Boundary> {
        Ok(match s {
            "clamped" => Boundary::Clamped,
            "extrapolate" => Boundary::Extrapolate,
            _ => {
                return spec(format!(
                    "boundary must be clamped or extrapolate, not {s:?}"
                ));
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct JacobianConfig {
    /// Image units per field sample along each spatial axis, in axis order.
    pub grid_spacing: Vec<f64>,
    pub min_det: f64,
    pub tight_error_bound: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub axis_roles: Vec<Role>,
    pub error_bound: f64,
    pub face_error_bound: Option<f64>,
    pub predictor: Predictor,
    pub boundary: Boundary,
    pub jacobian: Option<JacobianConfig>,
}

pub const INTERIOR: u8 = 0;
pub const FACE: u8 = 1;
pub const TIGHT: u8 = 2;
pub const VERBATIM: u8 = 3;

impl Config {
    /// Defaults: ENO predictor, clamped boundary, faces at `error_bound / 4`, no Jacobian.
    pub fn new(axis_roles: Vec<Role>, error_bound: f64) -> Config {
        Config {
            axis_roles,
            error_bound,
            face_error_bound: None,
            predictor: Predictor::Eno,
            boundary: Boundary::Clamped,
            jacobian: None,
        }
    }

    pub fn eps_face(&self) -> f64 {
        self.face_error_bound.unwrap_or(self.error_bound / 4.0)
    }

    pub fn eps_tight(&self) -> f64 {
        match &self.jacobian {
            Some(JacobianConfig {
                tight_error_bound: Some(t),
                ..
            }) => *t,
            _ => self.error_bound / 8.0,
        }
    }

    /// Absolute error bound per class: interior, face, tight, verbatim.
    pub fn eps_table(&self) -> [f64; 4] {
        [self.error_bound, self.eps_face(), self.eps_tight(), 0.0]
    }

    pub fn validate(&self) -> Result<()> {
        let roles = &self.axis_roles;
        let n_comp = roles.iter().filter(|r| **r == Role::Component).count();
        if n_comp > 1 {
            return spec("at most one axis may have role 'component'");
        }
        if !roles.iter().any(|r| r.predicted()) {
            return spec("at least one axis must be 'spatial' or 'smooth'");
        }
        if !(self.error_bound > 0.0 && self.error_bound.is_finite()) {
            return spec("error_bound must be a positive finite number");
        }
        if let Some(f) = self.face_error_bound
            && !(f > 0.0 && f <= self.error_bound)
        {
            return spec("face_error_bound must be in (0, error_bound]");
        }
        if let Some(j) = &self.jacobian {
            let ns = roles.iter().filter(|r| **r == Role::Spatial).count();
            if n_comp == 0 {
                return spec("jacobian needs a 'component' axis");
            }
            if !(1..=3).contains(&ns) {
                return spec("jacobian supports 1 to 3 spatial axes");
            }
            if j.grid_spacing.len() != ns {
                return spec(format!(
                    "jacobian.grid_spacing needs {ns} entries, one per spatial axis"
                ));
            }
            #[allow(clippy::neg_cmp_op_on_partial_ord)] // rejects NaN too
            if j.grid_spacing.iter().any(|s| !(*s > 0.0)) {
                return spec("jacobian.grid_spacing entries must be positive");
            }
            if let Some(t) = j.tight_error_bound
                && !(t > 0.0 && t <= self.error_bound)
            {
                return spec("tight_error_bound must be in (0, error_bound]");
            }
        }
        Ok(())
    }

    /// Parse the codec configuration from JSON. Accepts either the
    /// `configuration` object itself or the full `{"name": "warpfield",
    /// "configuration": {...}}` codec entry.
    #[cfg(feature = "json")]
    pub fn from_json(s: &str) -> Result<Config> {
        use serde_json::Value;
        let v: Value = serde_json::from_str(s)
            .map_err(|e| crate::Error::Spec(format!("invalid JSON: {e}")))?;
        let mut obj = match v {
            Value::Object(o) => o,
            _ => return spec("configuration must be a JSON object"),
        };
        if obj.contains_key("configuration") {
            if let Some(n) = obj.get("name")
                && n != "warpfield"
            {
                return spec(format!("codec name must be \"warpfield\", not {n}"));
            }
            obj = match obj.remove("configuration") {
                Some(Value::Object(o)) => o,
                _ => return spec("configuration must be a JSON object"),
            };
        }
        let num = |v: &Value, k: &str| -> Result<f64> {
            v.as_f64()
                .ok_or_else(|| crate::Error::Spec(format!("{k} must be a number")))
        };
        let string = |v: &Value, k: &str| -> Result<String> {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| crate::Error::Spec(format!("{k} must be a string")))
        };
        let mut cfg = Config::new(Vec::new(), 0.25);
        let mut have_roles = false;
        for (k, v) in obj.iter() {
            match k.as_str() {
                "axis_roles" => {
                    let a = v
                        .as_array()
                        .ok_or_else(|| crate::Error::Spec("axis_roles must be a list".into()))?;
                    cfg.axis_roles = a
                        .iter()
                        .map(|r| Role::parse(&string(r, "axis_roles entry")?))
                        .collect::<Result<_>>()?;
                    have_roles = true;
                }
                "error_bound" => cfg.error_bound = num(v, k)?,
                "face_error_bound" => {
                    cfg.face_error_bound = if v.is_null() { None } else { Some(num(v, k)?) }
                }
                "predictor" => cfg.predictor = Predictor::parse(&string(v, k)?)?,
                "boundary" => cfg.boundary = Boundary::parse(&string(v, k)?)?,
                "jacobian" => {
                    if v.is_null() {
                        cfg.jacobian = None;
                        continue;
                    }
                    let o = v
                        .as_object()
                        .ok_or_else(|| crate::Error::Spec("jacobian must be an object".into()))?;
                    let mut j = JacobianConfig {
                        grid_spacing: Vec::new(),
                        min_det: 0.0,
                        tight_error_bound: None,
                    };
                    let mut have_gs = false;
                    for (jk, jv) in o.iter() {
                        match jk.as_str() {
                            "grid_spacing" => {
                                let a = jv.as_array().ok_or_else(|| {
                                    crate::Error::Spec("grid_spacing must be a list".into())
                                })?;
                                j.grid_spacing = a
                                    .iter()
                                    .map(|x| num(x, "grid_spacing entry"))
                                    .collect::<Result<_>>()?;
                                have_gs = true;
                            }
                            "min_det" => j.min_det = num(jv, jk)?,
                            "tight_error_bound" => {
                                j.tight_error_bound = if jv.is_null() {
                                    None
                                } else {
                                    Some(num(jv, jk)?)
                                }
                            }
                            _ => return spec(format!("unknown jacobian key {jk:?}")),
                        }
                    }
                    if !have_gs {
                        return spec("jacobian.grid_spacing is required");
                    }
                    cfg.jacobian = Some(j);
                }
                _ => return spec(format!("unknown configuration key {k:?}")),
            }
        }
        if !have_roles {
            return spec("axis_roles is required");
        }
        cfg.validate()?;
        Ok(cfg)
    }
}
