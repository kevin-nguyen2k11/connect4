#ifndef C___CONNECT_4_RANDOM_H
#define C___CONNECT_4_RANDOM_H

#include <stdint.h>
#include <random>
#include <limits>
#include <vector>

namespace Random
{
    template<typename xtype>
    class Generator {
    public:
        typedef xtype result_type;

        explicit Generator(result_type seed)
        {
            xState = yState = zState = seed;
            this->operator()();
        }

        result_type operator()()
        {
            result_type xp = xState, yp = yState, zp = zState;
            xState = 15241094284759029579u * zp;
            yState = yp - xp;  yState = rotl(yState,12);
            zState = zp - yp;  zState = rotl(zState,44);
            return xp;
        }

        static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }

        static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }

    private:
        result_type rotl(result_type d, unsigned lrot)
        {
            return (d<<lrot) | (d>>(8*sizeof(d)-lrot));
        }

        result_type xState;
        result_type yState;
        result_type zState;
    };

    inline Generator generator{std::random_device{}()};

    inline int get(int min, int max)
    {
        return std::uniform_int_distribution{min,max}(generator);
    }
}

#endif //C___CONNECT_4_RANDOM_H
