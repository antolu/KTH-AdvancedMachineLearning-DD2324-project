# dd2434-project
String subsequence kernel (SSK) for text classification and its approximation

# C++ kernel

To compile the C++ kernel, first install the following packages
```
sudo apt-get install libboost-python-dev build-essential
```

Then install Boost
```
wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
cd boost_1_68_0
./bootstrap.sh
sudo ./b2 install
```

Compile kernel file with 
```
g++ -fPIC -std=c++11 -I /usr/include/python2.7/ -shared -o ssk.so ssk.cpp -lboost_python -lpython2.7
```

Usage as 
```
import ssk
ssk.SSK(doc1,doc2,lambda,n)
```