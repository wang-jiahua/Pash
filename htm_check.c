// gcc htm_check.c -o htm_check && ./htm_check
#include <stdio.h>
#include <stdint.h>

#if defined(_MSC_VER)
#   include <intrin.h>
#endif

void get_cpuid(uint32_t eax, uint32_t ecx, uint32_t *abcd){
    #if defined(_MSC_VER)
        __cpuidex(abcd,eax,ecx);
    #else
        uint32_t ebx,edx;
        #if defined( __i386__ ) && defined ( __PIC__ )
            /*in case of PIC, under 32-bit EBX cannot be clobbered*/
            __asm__( "movl %%ebx, %%edi \n\t xchgl %%ebx, %%edi" : "=D"(ebx),
        #else
            __asm__( "cpuid" : "+b"(ebx),
        #endif
            "+a"(eax), "+c"(ecx), "=d"(edx));

        abcd[0]=eax;abcd[1]=ebx;abcd[2]=ecx;abcd[3]=edx;
    #endif
}

int has_RTM_support(){
    uint32_t abcd[4];
    
    /*processor supports RTM execution if CPUID.07H.EBX.RTM [bit 11] = 1*/
    get_cpuid(0x7,0x0,abcd);
    return (abcd[1] & (1 << 11)) != 0;
}


int main(int argc, char **argv){
    
    if(has_RTM_support()){
        printf("This CPU supports RTM.");
    }else{
        printf("This CPU does NOT support RTM.");
    }
    return 0;
}