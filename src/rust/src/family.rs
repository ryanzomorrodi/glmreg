use extendr_api::prelude::*;
use ndarray::{Array1, ArrayBase, Data, Ix1};

pub enum ExpFamily {
    Gaussian,
    Poisson,
}

impl ExpFamily {
    pub fn link<S: Data<Elem = f64>>(&self, mu: &ArrayBase<S, Ix1>) -> Array1<f64> {
        match self {
            ExpFamily::Gaussian => mu.to_owned(),
            ExpFamily::Poisson => mu.mapv(|mu_i| mu_i.ln()),
        }
    }

    pub fn inv_link<S: Data<Elem = f64>>(&self, eta: &ArrayBase<S, Ix1>) -> Array1<f64> {
        match self {
            ExpFamily::Gaussian => eta.to_owned(),
            ExpFamily::Poisson => eta.mapv(|eta_i| eta_i.exp()),
        }
    }

    pub fn link_derivative<S: Data<Elem = f64>>(&self, mu: &ArrayBase<S, Ix1>) -> Array1<f64> {
        match self {
            ExpFamily::Gaussian => Array1::from_elem(mu.len(), 1.0),
            ExpFamily::Poisson => mu.mapv(|mu_i| 1.0 / mu_i),
        }
    }

    pub fn variance<S: Data<Elem = f64>>(&self, mu: &ArrayBase<S, Ix1>) -> Array1<f64> {
        match self {
            ExpFamily::Gaussian => Array1::from_elem(mu.len(), 1.0),
            ExpFamily::Poisson => mu.to_owned(),
        }
    }

    pub fn deviance<SY: Data<Elem = f64>, SM: Data<Elem = f64>>(
        &self,
        y: &ArrayBase<SY, Ix1>,
        mu: &ArrayBase<SM, Ix1>,
    ) -> f64 {
        match self {
            ExpFamily::Gaussian => y
                .iter()
                .zip(mu.iter())
                .map(|(y_i, mu_i)| (y_i - mu_i).powi(2))
                .sum(),
            ExpFamily::Poisson => {
                let mut dev = 0.0;
                for (&y_i, &mu_i) in y.iter().zip(mu.iter()) {
                    let log_diff = if y_i == 0.0 { 0.0 } else { (y_i / mu_i).ln() };
                    dev += y_i * log_diff - (y_i - mu_i);
                }

                dev * 2.0
            }
        }
    }

    pub fn initial_mu<S: Data<Elem = f64>>(&self, y: &ArrayBase<S, Ix1>) -> Array1<f64> {
        match self {
            _ => {
                let y_mean = y.mean().unwrap();
                y.mapv(|y_i| (y_mean + y_i) / 2.0)
            }
        }
    }
}
