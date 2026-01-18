use crate::spec::GLMFitOptions;
use extendr_api::prelude::*;
use ndarray::{Array1, Data};
use std::fmt;

pub struct GLMModel<S>
where
    S: Data<Elem = f64>,
{
    pub coefficients: Array1<f64>,
    pub coefficient_names: Option<Vec<String>>,
    pub outcome_name: Option<String>,
    pub fitted_values: Array1<f64>,
    pub residuals: Array1<f64>,
    pub null_deviance: f64,
    pub residual_deviance: f64,
    pub null_df: u32,
    pub residual_df: u32,
    pub iterations: u32,
    pub fit_options: GLMFitOptions<S>,
}

impl<S> From<GLMModel<S>> for Robj
where
    S: Data<Elem = f64>,
{
    fn from(model: GLMModel<S>) -> Self {
        let list = list!(
            coefficients = model.coefficients.to_vec(),
            residuals = model.residuals.to_vec(),
            fitted_values = model.fitted_values.to_vec()
        );

        list.into()
    }
}

impl<S> fmt::Display for GLMModel<S>
where
    S: Data<Elem = f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GLM Model Fit Results")?;
        writeln!(f, "=====================")?;

        let outcome = self.outcome_name.as_deref().unwrap_or("Y");
        let formula = if let Some(names) = &self.coefficient_names {
            names.join(" + ")
        } else {
            // Default to x0 + x1 + ... if names are missing
            (0..self.coefficients.len())
                .map(|i| format!("x{}", i))
                .collect::<Vec<_>>()
                .join(" + ")
        };
        writeln!(f, "Formula: {} ~ {}", outcome, formula)?;

        writeln!(f, "\nCoefficients:")?;
        writeln!(f, "{:<20} {:>12}", "Term", "Estimate")?;

        if let Some(names) = &self.coefficient_names {
            for (name, &coef) in names.iter().zip(self.coefficients.iter()) {
                writeln!(f, "{:<20} {:>12.6}", name, coef)?;
            }
        } else {
            for (i, &coef) in self.coefficients.iter().enumerate() {
                writeln!(f, "x{:<19} {:>12.6}", i, coef)?;
            }
        }

        writeln!(f, "\nDeviance Statistics:")?;
        writeln!(
            f,
            "    Null Deviance:     {:>12.4} on {} degrees of freedom",
            self.null_deviance, self.null_df
        )?;
        writeln!(
            f,
            "    Residual Deviance: {:>12.4} on {} degrees of freedom",
            self.residual_deviance, self.residual_df
        )?;

        // Convergence Info
        writeln!(f, "\nIterations: {}", self.iterations)?;

        Ok(())
    }
}
