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

gsave=function(x,x.new,y,ytype,ex,ey,comx,comy,r){
  n=dim(x)[1]
  kx0=gram.gauss(x,x,comx);kx=rbind(1,kx0)
  if(ytype=="scalar") {ky0=gram.gauss(y,y,comy);ky=rbind(1,ky0)} 
  if(ytype=="categorical") {ky0=gram.dis(y);ky=rbind(1,ky0)} 
  Q=diag(n)-rep(1,n)%*%t(rep(1,n))/n
  kkx=kx%*%Q%*%t(kx)
  kky=ky%*%t(ky)
  if(ytype=="scalar") kkyinv=ridgepower(sym(kky),ey,-1) 
  if(ytype=="categorical") kkyinv=matpower(sym(kky),-1) 
  piy=t(ky)%*%kkyinv%*%ky 
  sumlam=diag(apply(piy,1,sum))-piy%*%piy 
  a1=diag(apply(piy*piy,1,sum))-piy%*%piy/n 
  a2=(piy*piy)%*%piy-piy%*%diag(apply(piy,1,mean))%*%piy 
  a3=piy%*%diag(diag(piy%*%Q%*%piy))%*%piy 
  mid=Q/n-(2/n)*Q%*%sumlam%*%Q+Q%*%(a1-a2-t(a2)+a3)%*%Q 
  kx.new.0=gram.gauss(x,x.new,comx)
  kx.new=rbind(1,kx.new.0) 
  n1=dim(kx.new)[2];Q1=diag(n1)-rep(1,n1)%*%t(rep(1,n1))/n1 
  kk=ridgepower(kx%*%Q%*%t(kx),epsx,-1/2)%*%kx%*%Q 
  kk.new=ridgepower(kx%*%Q%*%t(kx),epsx,-1/2)%*%kx.new%*%Q1 
  pred=t(kk.new)%*%eigen(sym(kk%*%mid%*%t(kk)))$vectors[,1:r] 
  return(pred)
}

ridgepower=function(a,e,c){
  return(matpower(a+e*onorm(a)*diag(dim(a)[1]),c))}

# compute the operator norm (the largest eigenvalue) of a symmetric matrix
onorm=function(a) return(eigen(round((a+t(a))/2,8))$values[1])


matpower = function(a,alpha){
  a = round((a + t(a))/2,7); tmp = eigen(a) 
  return(tmp$vectors%*%diag((tmp$values)^alpha)%*%t(tmp$vectors))}

# symmetrize a matrix, in the situation where a matrix is theoretically symmetric but is numerically asymmtric due to numerical error
sym=function(a) return(round((a+t(a))/2,9))

# load data
data_name = "waveform"
reduce_dim = 10
mis_rec = c()
epsx = 0.01
for (cross_index in 1:20) {
  filename = paste0('./data/', data_name,'/',data_name,'_train_data_', cross_index, '.asc')
  x_train = read.table(file = filename, header = FALSE)
  x_train = as.matrix(x_train)
  filename = paste0('./data/', data_name,'/',data_name,'_train_labels_', cross_index, '.asc')
  y_train = read.table(file = filename, header = FALSE)
  y_train = (as.matrix(y_train) + 1) / 2
  filename = paste0('./data/', data_name,'/',data_name,'_test_data_', cross_index, '.asc')
  x_test = read.table(file = filename, header = FALSE)
  x_test = as.matrix(x_test)
  filename = paste0('./data/', data_name,'/',data_name,'_test_labels_', cross_index, '.asc')
  y_test = read.table(file = filename, header = FALSE)
  y_test = (as.matrix(y_test) + 1) / 2
  
  
  reduced_x_train = gsave(x_train, x_train, y_train, "categorical", 0.1, 0.1, 1, 1, reduce_dim)
  reduced_x_test = gsave(x_train, x_test, y_train, "categorical", 0.1, 0.1, 1, 1, reduce_dim)
  
  # fit a logistic regression model
  data = data.frame(x=reduced_x_train, y=y_train)
  model <- glm(V1 ~ ., data = data, family = "binomial")
  
  newdata <- data.frame(x=reduced_x_train)
  probabilities <- predict(model, newdata, type = "response")
  predicted.classes <- ifelse(probabilities < 0.5, 0, 1)
  print(paste0("train mis rate: ", 1 - sum(predicted.classes == y_train) / dim(y_train)[1]))
  
  newdata <- data.frame(x=reduced_x_test)
  probabilities <- predict(model, newdata, type = "response")
  predicted.classes <- ifelse(probabilities < 0.5, 0, 1)
  mis_rate = 1-sum(predicted.classes == y_test) / dim(y_test)[1]
  print(paste0("test mis rate: ", mis_rate))
  mis_rec[cross_index] = mis_rate
}

# save the results
filename = paste0('./result/GSAVE_IDA/', data_name,'/mis_rec_reduce_dim_', reduce_dim, '.csv')
write.csv(mis_rec, filename)

