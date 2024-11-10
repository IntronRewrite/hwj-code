#pragma once
#include<string>
#include"Gun.h"
using namespace std;

class Soldier{
    public:
        Soldier(string name);
        ~Soldier();
        void addGun(Gun *ptr_gun);
        void addBulletToGun(int num);
        bool fire();


    private:
        string _name;
        Gun *_ptr_gun;

};
///home/lw/hwj-code-c/7/src/Soldier.cpp