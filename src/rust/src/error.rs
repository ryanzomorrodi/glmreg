use std::{error, fmt};

#[derive(Debug)]
pub enum GLMFitError {
    NoPredictors,
    NoOutcome,
    NoFamily,
    ModelFittingError,
}

impl error::Error for GLMFitError {}

impl fmt::Display for GLMFitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GLMFitError::NoPredictors => write!(f, "Cannot fit GLM without predictors"),
            GLMFitError::NoOutcome => write!(f, "Cannot fit GLM without an outcome"),
            GLMFitError::NoFamily => write!(f, "Cannot fit GLM without a family"),
            GLMFitError::ModelFittingError => write!(f, "Something went wrong"),
        }
    }
}
