library(Matrix)
library(powerplus)

gsir=function(x,x.new,y,ytype,atype,ex,ey,complex.x,complex.y,r){ 
  n=dim(x)[1];p=dim(x)[2];Q=diag(n)-rep(1,n)%*%t(rep(1,n))/n 
  Kx=gram.gauss(x,x,complex.x)
  if(ytype=="scalar") Ky=gram.gauss(y,y,complex.y) 
  if(ytype=="categorical") Ky=gram.dis(y)
  Gx=Q%*%Kx%*%Q;Gy=Q%*%Ky%*%Q 
  Gxinv=matpower(sym(Gx+ex*onorm(Gx)*diag(n)),-1) 
  if(ytype=="categorical") Gyinv=matpower(sym(Gy),-1) 
  if(ytype=="scalar") Gyinv=matpower(sym(Gy+ey*onorm(Gy)*diag(n)),-1) 
  a1=Gxinv%*%Gx
  if(atype=="identity") a2=Gy 
  if(atype=="Gyinv") a2=Gy%*%Gyinv 
  gsir=a1%*%a2%*%t(a1) 
  v=eigen(sym(gsir))$vectors[,1:r] 
  Kx.new=gram.gauss(x,x.new,complex.x) 
  pred.new=t(t(v)%*%Gxinv%*%Q%*%Kx.new) 
  return(pred.new)
}

# compute the gram matrix based on the Gaussian radial basis kernel
gram.gauss=function(x,x.new,complexity){ 
  x=as.matrix(x);x.new=as.matrix(x.new) 
  n=dim(x)[1];m=dim(x.new)[1] 
  k2=x%*%t(x);k1=t(matrix(diag(k2),n,n));k3=t(k1);k=k1-2*k2+k3 
  sigma=sum(sqrt(k))/(2*choose(n,2));gamma=complexity/(2*sigma^2) 
  k.new.1=matrix(diag(x%*%t(x)),n,m)
  k.new.2=x%*%t(x.new) 
  k.new.3=matrix(diag(x.new%*%t(x.new)),m,n) 
  return(exp(-gamma*(k.new.1-2*k.new.2+t(k.new.3)))) 
}

# compute the discrete kernel
gram.dis=function(y){ 
  n=length(y);yy=matrix(y,n,n);diff=yy-t(yy);vecker=rep(0,n^2) 
  vecker[c(diff)==0]=1;vecker[c(diff)!=0]=0 
  return(matrix(vecker,n,n))
}

# compute the operator norm (the largest eigenvalue) of a symmetric matrix
onorm=function(a) return(eigen(round((a+t(a))/2,8))$values[1])

# symmetrize a matrix, in the situation where a matrix is theoretically symmetric but is numerically asymmtric due to numerical error
sym=function(a) return(round((a+t(a))/2,9))


matpower = function(a,alpha){
  a = round((a + t(a))/2,7); tmp = eigen(a) 
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}

# k-fold CV
cv.kfold=function(x,y,k,complex.x,ex){ 
  x=as.matrix(x);y=as.matrix(y);n=dim(x)[1] 
  ind=numeric();for(i in 1:(k-1)) ind=c(ind,floor(n/k)) 
  ind[k]=n-floor(n*(k-1)/k)
  cv.out=0
  for(i in 1:k){
    if(i<k) groupi=((i-1)*floor(n/k)+1):(i*floor(n/k)) 
    if(i==k) groupi=((k-1)*floor(n/k)+1):n 
    groupic=(1:n)[-groupi] 
    x.tra=as.matrix(x[groupic,]);y.tra=as.matrix(y[groupic,]) 
    x.tes=as.matrix(x[groupi,]);y.tes=as.matrix(y[groupi,]) 
    Kx=gram.gauss(x.tra,x.tra,complex.x) 
    Kx.tes=gram.gauss(x.tra,x.tes,complex.x) 
    Ky=gram.gauss(y.tra,y.tra,1) 
    Ky.tes=gram.gauss(y.tra,y.tes,1)
    cvi=sum((t(Ky.tes)-t(Kx.tes)%*% 
               matpower(Kx+ex*onorm(Kx)*diag(dim(y.tra)[1]),-1)%*%Ky)^2)
    cv.out=cv.out+cvi 
  }
  return(cv.out)
}

# load data
data_name = "covtype"
reduce_dim = 25
mis_rec = c()
for (cross_index in 1:10) {
  filename = paste0('./data/covtype/x_train_cross_',cross_index,'.txt')
  x_train = read.csv(file = filename)
  x_train = as.matrix(x_train)
  filename = paste0('./data/covtype/y_train_cross_',cross_index,'.txt')
  y_train = read.table(file = filename)
  filename = paste0('./data/covtype/x_test_cross_',cross_index,'.txt')
  x_test = read.table(file = filename)
  x_test = as.matrix(x_test)
  filename = paste0('./data/covtype/y_test_cross_',cross_index,'.txt')
  y_test = read.table(file = filename)
  
  
  reduced_x_train = gsir(x_train, x_train, y_train, "categorical", "identity", 0.1, 0.1, 1, 1, reduce_dim)
  reduced_x_test = gsir(x_train, x_test, y_train, "categorical", "identity", 0.1, 0.1, 1, 1, reduce_dim)
  
  filename = paste0("./result/covtype/GSIR/cross_",cross_index,"/reduced_dim_25_x_train.txt")
  write.csv(reduced_x_train, filename)
  filename = paste0("./result/covtype/GSIR/cross_",cross_index,"/reduced_dim_25_x_test.txt")
  write.csv(reduced_x_test, filename)
}

# save the results
filename = paste0('./result/GSIR_IDA/', data_name,'/mis_rec_reduce_dim_', reduce_dim, '.csv')
write.csv(mis_rec, filename)
