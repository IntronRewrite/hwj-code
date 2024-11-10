#pragma once
#include<iostream>

class swap{
    public:
        swap(int a,int b){
            this->a=a;
            this->b=b;
        }
        void run();
        void printInfo();

    private:
        int a;
        int b;
};