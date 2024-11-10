#include"swap.h"
int main(int arg,char **argv){
    swap myswap(10,20);
    myswap.printInfo();
    myswap.run();
    myswap.printInfo();

    return 0;
}