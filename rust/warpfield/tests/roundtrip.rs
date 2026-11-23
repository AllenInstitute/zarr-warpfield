use warpfield::{
    Boundary, Config, EncodeOptions, Error, JacobianConfig, Predictor, Role, decode, encode,
};

/// Smooth 3D field with a 6-voxel slip step, shape (Z, Y, X, 3).
fn field(nz: usize, ny: usize, nx: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(nz * ny * nx * 3);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let (zf, yf, xf) = (z as f32, y as f32, x as f32);
                let step = if xf + 0.5 * yf > 20.0 { 6.0 } else { 0.0 };
                v.push(2.0 * (zf / 9.0).sin() + 0.03 * xf + step);
                v.push(1.5 * (xf / 7.0).cos() - 0.02 * zf);
                v.push(0.5 * ((yf + zf) / 11.0).sin());
            }
        }
    }
    v
}

fn roles3() -> Vec<Role> {
    vec![Role::Spatial, Role::Spatial, Role::Spatial, Role::Component]
}

#[test]
fn bounds_hold_for_every_predictor_and_boundary() {
    let shape = [13, 17, 21, 3];
    let x = field(13, 17, 21);
    for predictor in [Predictor::Linear, Predictor::Cubic, Predictor::Eno] {
        for boundary in [Boundary::Clamped, Boundary::Extrapolate] {
            let mut cfg = Config::new(roles3(), 0.25);
            cfg.predictor = predictor;
            cfg.boundary = boundary;
            let opts = EncodeOptions {
                classes: true,
                ..Default::default()
            };
            let enc = encode(&x, &shape, &cfg, &opts).unwrap();
            let y: Vec<f32> = decode(&enc.bytes, &shape, &cfg).unwrap();
            let cls = enc.classes.unwrap();
            for i in 0..x.len() {
                let bound = cfg.eps_table()[cls[i] as usize];
                assert!(
                    (x[i] - y[i]).abs() as f64 <= bound,
                    "{predictor:?} {boundary:?} {i}"
                );
            }
            assert!(enc.bytes.len() * 4 < x.len() * 4, "should compress");
        }
    }
}

#[test]
fn float64_and_jacobian() {
    let shape = [9, 10, 11, 3];
    let x: Vec<f64> = field(9, 10, 11).into_iter().map(f64::from).collect();
    let mut cfg = Config::new(roles3(), 0.5);
    cfg.jacobian = Some(JacobianConfig {
        grid_spacing: vec![2.0, 2.0, 2.0],
        min_det: 0.1,
        tight_error_bounds: Vec::new(),
    });
    let enc = encode(&x, &shape, &cfg, &EncodeOptions::default()).unwrap();
    let y: Vec<f64> = decode(&enc.bytes, &shape, &cfg).unwrap();
    assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() <= 0.5));
}

#[test]
fn json_config() {
    let c = Config::from_json(
        r#"{"name": "warpfield", "configuration": {"axis_roles": ["smooth", "spatial",
        "spatial", "component"], "error_bound": 0.1, "face_error_bound": 0.025,
        "predictor": "cubic", "jacobian": {"grid_spacing": [4, 4], "min_det": 0.2}}}"#,
    )
    .unwrap();
    assert_eq!(c.axis_roles[0], Role::Smooth);
    assert_eq!(c.eps_table()[..4], [0.1, 0.025, 0.1 / 8.0, 0.0]);
    let ladder = Config::from_json(
        r#"{"axis_roles": ["spatial", "spatial", "component"], "error_bound": 0.1,
        "jacobian": {"grid_spacing": [4, 4], "tight_error_bound": [0.02, 0.005]}}"#,
    )
    .unwrap();
    assert_eq!(ladder.eps_table(), [0.1, 0.025, 0.02, 0.0, 0.005, 0.0, 0.0]);
    for bad in [
        r#"{"error_bound": 0.1}"#,
        r#"{"axis_roles": ["spatial"], "error_bound": -1}"#,
        r#"{"axis_roles": ["spatial"], "surprise": 1}"#,
        r#"{"axis_roles": ["spatial", "component"], "jacobian": {"grid_spacing": [1, 1]}}"#,
        r#"{"axis_roles": ["component", "component", "spatial"]}"#,
        r#"{"axis_roles": ["spatial", "component"], "jacobian": {"grid_spacing": [1], "tight_error_bound": [0.01, 0.02]}}"#,
        r#"{"axis_roles": ["spatial", "component"], "jacobian": {"grid_spacing": [1], "tight_error_bound": []}}"#,
    ] {
        assert!(
            matches!(Config::from_json(bad), Err(Error::Spec(_))),
            "{bad}"
        );
    }
}

#[test]
fn malformed_chunks_are_errors() {
    let shape = [8, 8, 8, 3];
    let x = field(8, 8, 8);
    let cfg = Config::new(roles3(), 0.25);
    let b = encode(&x, &shape, &cfg, &EncodeOptions::default())
        .unwrap()
        .bytes;
    for bad in [&b[..10], &b[..b.len() - 1], &b"nope"[..]] {
        assert!(matches!(
            decode::<f32>(bad, &shape, &cfg),
            Err(Error::Format(_))
        ));
    }
    assert!(decode::<f32>(&b, &[8, 8, 9, 3], &cfg).is_err());
    let mut flipped = b.clone();
    let k = flipped.len() / 2;
    flipped[k] ^= 0xff;
    let _ = decode::<f32>(&flipped, &shape, &cfg); // must not panic
}
