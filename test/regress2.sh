../rrt -d 0 -i ../scenes/final.txt -s 16 -2 -o final2_16.png   2>  regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 32 -2 -o final2_32.png   2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 64 -2 -o final2_64.png   2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 128 -2 -o final2_128.png 2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 256 -2 -o final2_256.png 2>> regress2.log

../rrt -d 0 -i ../scenes/final.txt -s 16 -tx 16 -ty 16 -o final_16.png       2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 32 -tx 16 -ty 16 -o final_32.png       2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 64 -tx 16 -ty 16 -o final_64.png       2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 128 -tx 16 -ty 16 -o final_128.png     2>> regress2.log
../rrt -d 0 -i ../scenes/final.txt -s 256 -tx 16 -ty 16 -o final_256.png     2>> regress2.log
