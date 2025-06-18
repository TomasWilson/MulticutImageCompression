#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>

extern std::unordered_map<int, std::chrono::system_clock::time_point> times;
void tic(int marker = 0); // measure the current time and store it w.r.t the marker
int toc(const std::string& msg = "", int marker=0); // display message and the duration in milliseconds
int toctic(const std::string& msg = "", int marker=0); // displays the duration and then resets the start time