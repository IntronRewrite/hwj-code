#include "Soldier.h"
using namespace std;

Soldier::Soldier(string name){
    this->_name = name;
    this->_ptr_gun = nullptr;
}

void Soldier::addGun(Gun *ptr_gun){
    this->_ptr_gun = ptr_gun;
}


void Soldier::addBulletToGun(int num){
    this->_ptr_gun->addBullet(num);
}

bool Soldier::fire(){
    
    return this->_ptr_gun->shoot();
}

Soldier::~Soldier(){
    if(this->_ptr_gun != nullptr){
        delete this->_ptr_gun;
    }
    this->_ptr_gun = nullptr;
}