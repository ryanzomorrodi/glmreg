test_that("multiplication works", {
  data(penguins)
  penguins <- na.omit(penguins)

  model <- simple_glm(bill_len ~ log(body_mass) + species, penguins)
})
