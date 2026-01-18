use crate::{error::GLMFitError, family::ExpFamily, model::GLMModel};
use extendr_api::prelude::ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2, Zip};
use ndarray_linalg::{Solve, QR};

pub struct GLMSpec<S>
where
    S: Data<Elem = f64>,
{
    parameters: Option<ArrayBase<S, Ix2>>,
    parameter_names: Option<Vec<String>>,
    outcome: Option<ArrayBase<S, Ix1>>,
    outcome_name: Option<String>,
    family: Option<ExpFamily>,
    fit_options: GLMFitOptions<S>,
}

impl<S> GLMSpec<S>
where
    S: Data<Elem = f64>,
{
    pub fn new() -> Self {
        GLMSpec {
            parameters: None,
            parameter_names: None,
            outcome: None,
            outcome_name: None,
            family: None,
            fit_options: GLMFitOptions::default(),
        }
    }

    pub fn parameters(
        mut self,
        parameter_names: Option<Vec<String>>,
        parameters: ArrayBase<S, Ix2>,
    ) -> Self {
        self.parameters = Some(parameters);
        self.parameter_names = parameter_names;
        self
    }

    pub fn outcome(mut self, outcome_name: Option<String>, outcome: ArrayBase<S, Ix1>) -> Self {
        self.outcome = Some(outcome);
        self.outcome_name = outcome_name;
        self
    }

    pub fn family(mut self, family: ExpFamily) -> Self {
        self.family = Some(family);
        self
    }

    pub fn fit_options(mut self, fit_options: GLMFitOptions<S>) -> Self {
        self.fit_options = fit_options;
        self
    }

    pub fn fit(self) -> Result<GLMModel<S>, GLMFitError> {
        let x = self.parameters.as_ref().ok_or(GLMFitError::NoPredictors)?;
        let y = self.outcome.as_ref().ok_or(GLMFitError::NoOutcome)?;
        let family = self.family.as_ref().ok_or(GLMFitError::NoFamily)?;
        let max_iter = self.fit_options.max_iter;

        let n_features = x.ncols();
        let n_obs = y.len();
        let null_mu = Array1::from_elem(n_obs, y.mean().unwrap());
        let null_deviance = family.deviance(y, &null_mu);
        let null_df = (n_obs - 1) as u32;

        let mut mu = self
            .fit_options
            .mu_start
            .as_ref()
            .map(|m| m.to_owned())
            .unwrap_or_else(|| family.initial_mu(y));
        let mut eta = family.link(&mu);
        let mut beta = self
            .fit_options
            .beta_start
            .as_ref()
            .map(|b| b.to_owned())
            .unwrap_or_else(|| Array1::<f64>::zeros(n_features));
        let mut deviance = family.deviance(y, &mu);

        let mut n_iter = 0;
        for iter in 1..=max_iter {
            n_iter = iter;
            let g_prime = family.link_derivative(&mu);
            let variance = family.variance(&mu);

            let z = &eta + &(y - &mu) * &g_prime;

            // W = 1/(Var(μ) * g'(μ)^2)
            // so, sqrt(W) = 1/(sqrt(Var(μ)) * g'(μ))
            let w_sqrt = Zip::from(&g_prime)
                .and(&variance)
                .map_collect(|g_prime_i, variance_i| 1.0 / (variance_i.sqrt() * g_prime_i.abs()));

            let x_weighted = x * &w_sqrt.clone().insert_axis(Axis(1));
            let z_weighted = &z * &w_sqrt;

            let (q, r) = x_weighted.qr().expect("QR decomposition failed");
            let mut qty = q.t().dot(&z_weighted);

            let beta_new = r.solve_inplace(&mut qty).expect("Linear solve of R failed");

            match self.step_halving(&beta, &beta_new, deviance) {
                Ok(outcome) => {
                    let is_converged = match outcome {
                        StepHalvingOutcome::Convergence(_) => true,
                        _ => false,
                    };
                    let (beta_new, mu_new, dev_new) = match outcome {
                        StepHalvingOutcome::Improvement(t) | StepHalvingOutcome::Convergence(t) => {
                            t
                        }
                    };

                    beta = beta_new;
                    mu = mu_new;
                    eta = x.dot(&beta);
                    deviance = dev_new;

                    if is_converged {
                        break;
                    }
                }
                Err(err) => return Err(err),
            }
        }

        let residuals = y - &mu;
        let residual_deviance = family.deviance(y, &mu);
        let residual_df = (n_obs - n_features) as u32;

        Ok(GLMModel {
            coefficients: beta,
            coefficient_names: self.parameter_names,
            outcome_name: self.outcome_name,
            fitted_values: mu,
            residuals,
            null_deviance,
            residual_deviance,
            null_df,
            residual_df,
            iterations: n_iter as u32,
            fit_options: self.fit_options,
        })
    }

    fn step_halving(
        &self,
        beta_old: &Array1<f64>,
        beta_new: &Array1<f64>,
        dev_old: f64,
    ) -> Result<StepHalvingOutcome<(Array1<f64>, Array1<f64>, f64)>, GLMFitError> {
        let x = self.parameters.as_ref().ok_or(GLMFitError::NoPredictors)?;
        let y = self.outcome.as_ref().ok_or(GLMFitError::NoOutcome)?;
        let family = self.family.as_ref().ok_or(GLMFitError::NoFamily)?;
        let epsilon = self.fit_options.epsilon;

        let mut step_size = 1.0;
        let max_halving = 10;

        for _ in 0..max_halving {
            let delta_beta = beta_new - beta_old;
            let beta_trial = beta_old + &(delta_beta * step_size);
            let eta_trial = x.dot(&beta_trial);
            let mu_trial = family.inv_link(&eta_trial);
            let dev_new = family.deviance(&y, &mu_trial);

            let rel_delta_dev = (dev_new - dev_old) / (0.1 + dev_new.abs());

            if rel_delta_dev.abs() < epsilon {
                return Ok(StepHalvingOutcome::Convergence((
                    beta_trial, mu_trial, dev_new,
                )));
            } else if rel_delta_dev >= epsilon {
                step_size *= 0.5;
                continue;
            }

            return Ok(StepHalvingOutcome::Improvement((
                beta_trial, mu_trial, dev_new,
            )));
        }

        Err(GLMFitError::ModelFittingError)
    }
}
enum StepHalvingOutcome<T> {
    Convergence(T),
    Improvement(T),
}

pub struct GLMFitOptions<S>
where
    S: Data<Elem = f64>,
{
    pub epsilon: f64,
    pub max_iter: u32,
    pub mu_start: Option<ArrayBase<S, Ix1>>,
    pub beta_start: Option<ArrayBase<S, Ix1>>,
}

impl<S> Default for GLMFitOptions<S>
where
    S: Data<Elem = f64>,
{
    fn default() -> Self {
        Self {
            epsilon: 1e-8,
            max_iter: 25,
            mu_start: None,
            beta_start: None,
        }
    }
}
