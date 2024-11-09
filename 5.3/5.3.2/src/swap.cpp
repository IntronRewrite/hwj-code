#include<iostream>
#include "swap.h"
using namespace std;

void swap::run(){
    int tmp;
    tmp = a;
    a = b;
    b = tmp;
}

void swap::printInfo(){
    cout<<"a = "<<a<<endl;
    cout<<"b = "<<b<<endl;
}