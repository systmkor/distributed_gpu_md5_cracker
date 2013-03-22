#!/bin/bash

# This script runs a series of tests comparing run times of 
# the CPU and GPU versions of our MD5 cracker

make clean

echo 'Compiling...'
make

if [[ -x ./md5gcracker && -x ./md5cracker && -x ./passgen ]]; then
    true
else
    echo 'Unable to find necessary executables'
    exit
fi

for x in 100000 1000000 10000000 40000000; do
    echo
    echo '================================================================='
    if [ -f /tmp/$x ]; then
        printf "%'0.f word dictionary found.\n" $x
    else
        printf "%'0.f word dictionary not found. Generating...\n" $x
        ./passgen $x > /tmp/$x
    fi
    echo "Dictionary file size: `ls -sh /tmp/$x | cut -d' ' -f1`"

    MID=$(($x/2))
    MD51=`tail -1 /tmp/$x | head -c8 | md5sum | head -c32`
    MD52=`head -1 /tmp/$x | head -c8 | md5sum | head -c32`
    MD53=`sed -n "$MID""p" /tmp/$x | head -c8 | md5sum | head -c32`

    TIME=$(/usr/bin/time -f "%e" ./md5gcracker /tmp/$x $MD52 2>&1 > out)
    echo "GPU: F, "$TIME"s, `cat out`"
    TIME=$(/usr/bin/time -f "%e" ./md5gcracker /tmp/$x $MD53 2>&1 > out)
    echo "GPU: M, "$TIME"s, `cat out`"
    GTIME=$(/usr/bin/time -f "%e" ./md5gcracker /tmp/$x $MD51 2>&1 > out)
    echo GPU: L, "$GTIME"s, `cat out`

    echo
    TIME=`/usr/bin/time -f "%e" ./md5cracker /tmp/$x $MD52 2>&1 > out`
    echo "CPU: F, "$TIME"s, `cat out`"
    TIME=`/usr/bin/time -f "%e" ./md5cracker /tmp/$x $MD53 2>&1 > out`
    echo "CPU: M, "$TIME"s, `cat out`"
    CTIME=`/usr/bin/time -f "%e" ./md5cracker /tmp/$x $MD51 2>&1 > out`
    echo CPU: L, "$CTIME"s, `cat out`, $(python -c "print str(round($CTIME/float($GTIME), 1))+'x'")
    rm -f out
done


