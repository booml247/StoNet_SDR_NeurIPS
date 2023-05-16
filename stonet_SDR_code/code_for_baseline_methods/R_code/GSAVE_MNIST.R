
# load data
load_image_file <- function(filename) {
  ret = list()
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  ret$n = readBin(f,'integer',n=1,size=4,endian='big')
  nrow = readBin(f,'integer',n=1,size=4,endian='big')
  ncol = readBin(f,'integer',n=1,size=4,endian='big')
  x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
  ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
  close(f)
  ret
}

load_label_file <- function(filename) {
  f = file(filename,'rb')
  readBin(f,'integer',n=1,size=4,endian='big')
  n = readBin(f,'integer',n=1,size=4,endian='big')
  y = readBin(f,'integer',n=n,size=1,signed=F)
  close(f)
  y
}

train <- load_image_file("./data/MNIST/raw/train-images-idx3-ubyte")
test <- load_image_file("./data/MNIST/raw/t10k-images-idx3-ubyte")

train$y <- load_label_file("./data/MNIST/raw/train-labels-idx1-ubyte")
test$y <- load_label_file("./data/MNIST/raw/t10k-labels-idx1-ubyte")

x_train <- train$x[1:20000,]
y_train <- train$y[1:20000]
x_test <- test$x

data_name = "MNIST"
reduce_dim = 196
mis_rec = c()



reduced_x_train = gsave(x_train, x_train, y_train, "categorical", 1e-6, 1e-6, 0.2, 0.2, reduce_dim)
start_time <- Sys.time()
reduced_x_test = gsave(x_train, x_test, y_train, "categorical", 1e-6, 1e-6, 0.2, 0.2, reduce_dim)
time_elapse <- Sys.time() - start_time

filename = paste0('./result/GSAVE_MNIST/', data_name,'/reduced_x_train_reduce_dim_', reduce_dim, '.csv')
write.csv(reduced_x_train, filename)
filename = paste0('./result/GSAVE_MNIST/', data_name,'/reduced_x_test_reduce_dim_', reduce_dim, '.csv')
write.csv(reduced_x_test, filename)
filename = paste0('./result/GSAVE_MNIST/', data_name,'/time_elapse_reduce_dim_', reduce_dim, '.csv')
write.csv(time_elapse, filename)
