
# if this gives error, try installing reshape2 package in R first

mf = function(input_path, output_path, dim, dataname){
  source('data/libpmf-1.41/R/libpmf.R')
  # path: path to data
  # output_path needs to end with "/"
  # dim: a str denoting the dimension of matrices
  mydata = read.table(input_path)
  row.idx = as.integer(mydata$V1)
  col.idx = as.integer(mydata$V2)
  obs.val = mydata$V3
  param = paste('-l 0.05 -k', dim, sep=" ")
  model = pmf.train.coo(row.idx, col.idx, obs.val, param.str=param)
  
  fn_movie = paste(output_path, dataname, 'movies_matrix_d', dim, sep="")
  fn_user = paste(output_path, dataname, 'users_matrix_d', dim, sep="")
  write.table(model$H, file=fn_movie, col.names = FALSE, row.names = FALSE)
  write.table(model$W, file=fn_user, col.names = FALSE, row.names = FALSE)
}
