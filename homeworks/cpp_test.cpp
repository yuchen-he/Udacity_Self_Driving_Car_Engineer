#include <iostream>
// #include eigen.h

int main()
{
    int i = 5;
    std::cout << "breakpoint!" << std::endl;
    while (i != 0)
    {
        i -= 1;
        std::cout << "i = " << i << std::endl;
    }
    
    return 0;
}