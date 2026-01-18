predict_simple_lm_numeric <- function(object, predictors) {
  coefs <- object$coefs
  pred <- as.vector(predictors %*% coefs)
  out <- hardhat::spruce_numeric(pred)

  out
}

predict_simple_lm_bridge <- function(type, object, predictors) {
  type <- rlang::arg_match(type, "numeric")
  predictors <- as.matrix(predictors)
  switch(
    type,
    numeric = predict_simple_lm_numeric(object, predictors)
  )
}

#' @export
predict.simple_lm <- function(object, new_data, type = "numeric", ...) {
  processed <- hardhat::forge(new_data, object$blueprint)
  out <- predict_simple_lm_bridge(type, object, processed$predictors)
  hardhat::validate_prediction_size(out, new_data)

  out
}
