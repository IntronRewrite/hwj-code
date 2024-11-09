#include<iostream>
#include "swap.h"
using namespace std;

int main(int argc,char **argv){
    int val1 =10;
    int val2 =20;

    cout<<"交换前：";
    cout<<"val1 = "<<val1;
    cout<<"val2 = "<<val2;
    swap(val1,val2);
    cout<<"交换后：";
    cout<<"val1 = "<<val1;
    cout<<"val2 = "<<val2;

}