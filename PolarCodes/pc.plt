set ls 1 dt 1 lc rgb "#FF0000" lw 1	   
set ls 2 dt 3 lc rgb "#880000" lw 2
set xtics 1
set mxtics 2
set mytics (5)
set logscale y 10
set grid xtics mxtics ytics mytics 
set xlabel "SNRb, dB"
set ylabel "FER"
plot [1:5] [0.0001:1] \
"pc.txt" using	1:3 title "SC (1024, 512)" with lines ls 1 ,\
"pc.txt" using	1:2 title "SC Minsum (1024, 512)" with lines ls 2 