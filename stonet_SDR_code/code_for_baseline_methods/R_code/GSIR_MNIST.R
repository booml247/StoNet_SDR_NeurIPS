
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
reduce_dim = 392
mis_rec = c()


start_time <- Sys.time()
reduced_x_train = gsir(x_train, x_train, y_train, "categorical", "identity", 0.1, 0.1, 1, 1, reduce_dim)
reduced_x_test = gsir(x_train, x_test, y_train, "categorical", "identity", 0.1, 0.1, 1, 1, reduce_dim)
time_elapse <- Sys.time() - start_time

filename <- "./result/MNIST/GSIR/reduced_x_train_reduce_dim_392.csv"
write.csv(reduced_x_train, filename)
filename <- "./result/MNIST/GSIR/reduced_x_test_reduce_dim_392.csv"
write.csv(reduced_x_test, filename)
filename <- "./result/MNIST/GSIR/time_elapse_reduce_dim_392.csv"
write.csv(time_elapse, filename)
