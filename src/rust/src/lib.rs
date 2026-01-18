mod error;
mod family;
mod model;
mod spec;

use crate::{
    family::ExpFamily,
    spec::{GLMFitOptions, GLMSpec},
};

use extendr_api::prelude::*;
use ndarray::{ArrayView1, ArrayView2};

/// @export
#[extendr]
fn glm_irls(x: Robj, y: Robj, family: String) -> Robj {
    let param_names: Vec<String> = x
        .names()
        .map(|names| names.map(|s| s.to_string()).collect())
        .unwrap_or_default();

    let x: ArrayView2<f64> = x.try_into().unwrap();
    let y: ArrayView1<f64> = y.try_into().unwrap();

    let fit_family = match family.as_str() {
        "gaussian" => ExpFamily::Gaussian,
        "poisson" => ExpFamily::Poisson,
        _ => todo!(),
    };

    let spec = GLMSpec::new()
        .parameters(Some(param_names), x)
        .outcome(None, y)
        .family(fit_family)
        .fit_options(GLMFitOptions {
            max_iter: 100,
            ..Default::default()
        });

    let model = spec.fit().unwrap();

    model.into_robj()
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod glmreg;
    fn glm_irls;
}
