#ifndef SERIALIZATION_H
#define SERIALIZATION_H

#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cmath>
#include <cassert>
#include <cstdlib>

#include "vector.H"

using namespace Foam;

std::vector<int> listToVector(std::list<int> l)
{
	std::vector<int> vec(l.size());
	int i=0;
	for(std::list<int>::iterator it=l.begin(); it != l.end(); it++) {
		vec.at(i) = (*it);
		i++;
	}
	return vec;
}

std::vector<scalar> listToVector(std::list<scalar> l)
{
	std::vector<scalar> vec(l.size());
	scalar i=0;
	for(std::list<scalar>::iterator it=l.begin(); it != l.end(); it++) {
		vec.at(i) = (*it);
		i++;
	}
	return vec;
}

std::vector<scalar> listToVector(std::list<vector> l)
{
	std::vector<scalar> vec(l.size()*4);
	int i=0;
	for(std::list<vector>::iterator it=l.begin(); it != l.end(); it++) {
		vec.at(i*4)     = (*it).x();
		vec.at(i*4 + 1) = (*it).y();
		vec.at(i*4 + 2) = (*it).z();
		i++;
	}
	return vec;
}

template<class T>
void writeListBinary(std::list<T> data, std::string filename)
{
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);

	if(!file.is_open()) {
		printf("Something went wrong when trying to open file %s.\n",
			filename.c_str());
		exit(-1);
	}

	char* c;

	for(
		typename std::list<T>::iterator it = data.begin();
		it != data.end();
		it++
	) {
		// Uhhh is there a better, less ugly way to do this?!
		c = static_cast<char*>(static_cast<void*>(&(*it)));
		file.write(c, sizeof(T));
	}
	
	file.close();
}

template<class T>
std::list<T> readListBinary(std::string filename)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);

	if(!file.is_open()) {
		printf("Something went wrong when trying to open file %s.\n",
			filename.c_str());
		exit(-1);
	}
	
	T tmp;
	std::list<T> dataList;
	
	while(file.read(static_cast<char*>(static_cast<void*>(&tmp)), sizeof(T))) {
		dataList.push_back(tmp);
	}

	return dataList;
}

void writeVectorListBinary(std::list<vector> data, std::string filename)
{
	std::fstream file(filename.c_str(), std::ios::out | std::ios::binary);

	if(!file.is_open()) {
		printf("Something went wrong when trying to open file %s.\n",
			filename.c_str());
		exit(-1);
	}

	char* c;

	for(
		std::list<vector>::iterator it = data.begin();
		it != data.end();
		it++
	) {
		c = static_cast<char*>(static_cast<void*>(&((*it).x())));
		file.write(c, sizeof(scalar));
		c = static_cast<char*>(static_cast<void*>(&((*it).y())));
		file.write(c, sizeof(scalar));
		c = static_cast<char*>(static_cast<void*>(&((*it).z())));
		file.write(c, sizeof(scalar));
	}
	
	file.close();
}

std::list<vector> readVectorListBinary(std::string filename)
{
	std::fstream file(filename.c_str(), std::ios::in | std::ios::binary);

	if(!file.is_open()) {
		printf("Something went wrong when trying to open file %s.\n",
			filename.c_str());
		exit(-1);
	}
	
	std::list<vector> dataList;
	
	vector tmp;
	while(file.read(static_cast<char*>(static_cast<void*>(&(tmp.x()))), sizeof(scalar))
	   && file.read(static_cast<char*>(static_cast<void*>(&(tmp.y()))), sizeof(scalar))
	   && file.read(static_cast<char*>(static_cast<void*>(&(tmp.z()))), sizeof(scalar))
	) {
		dataList.push_back(tmp);
	}

	return dataList;
}

#endif
