#' @export
simple_glm <- function(x, ...) {
  UseMethod("simple_glm")
}

#' @export
simple_glm.default <- function(x, ...) {
  stop(
    "`simple_glm()` is not defined for a '",
    class(x)[1],
    "'.",
    call. = FALSE
  )
}

#' @export
simple_glm.data.frame <- function(x, y, intercept = TRUE, ...) {
  blueprint <- hardhat::default_xy_blueprint(intercept = intercept)
  processed <- hardhat::mold(x, y, blueprint = blueprint)
  simple_glm_bridge(processed)
}

#' @export
simple_glm.matrix <- function(x, y, intercept = TRUE, ...) {
  blueprint <- hardhat::default_xy_blueprint(intercept = intercept)
  processed <- hardhat::mold(x, y, blueprint = blueprint)
  simple_glm_bridge(processed)
}

#' @export
simple_glm.formula <- function(formula, data, intercept = TRUE, ...) {
  blueprint <- hardhat::default_formula_blueprint(intercept = intercept)
  processed <- hardhat::mold(formula, data, blueprint = blueprint)
  simple_glm_bridge(processed)
}

#' @export
simple_glm.recipe <- function(x, data, intercept = TRUE, ...) {
  blueprint <- hardhat::default_recipe_blueprint(intercept = intercept)
  processed <- hardhat::mold(x, data, blueprint = blueprint)
  simple_glm_bridge(processed)
}


simple_glm_bridge <- function(processed) {
  hardhat::validate_outcomes_are_univariate(processed$outcomes)

  predictors <- as.matrix(processed$predictors)
  outcomes <- processed$outcomes[[1]]

  fit <- simple_glm_impl(predictors, outcomes)

  new_simple_glm(
    coefficients = fit$coefficients,
    residuals = fit$residuals,
    fitted_values = fit$fitted_values,
    blueprint = processed$blueprint
  )
}

simple_glm_impl <- function(predictors, outcomes) {
  result <- glm_irls(predictors, outcomes)
  names(result$coefficients) <- colnames(predictors)

  result
}

new_simple_glm <- function(coefficients, residuals, fitted_values, blueprint) {
  if (!is.numeric(coefficients)) {
    stop("`coefficients` should be a numeric vector.", call. = FALSE)
  }
  if (!is.numeric(residuals)) {
    stop("`residuals` should be a numeric vector.", call. = FALSE)
  }
  if (!is.numeric(fitted_values)) {
    stop("`fitted_values` should be a numeric vector.", call. = FALSE)
  }

  hardhat::new_model(
    coefficients = coefficients,
    residuals = residuals,
    fitted_values = fitted_values,
    blueprint = blueprint,
    class = "simple_glm"
  )
}

#' @export
print.simple_glm <- function(x, ...) {
  cat("Coefficients:\n")
  print(x$coefficients)
}
