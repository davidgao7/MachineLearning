touch need_delete.txt
yes | rm -i need_delete.txt # this will answer 'y' for us when delete the file

# getting input:
echo $'\n'Hello! What\'s your name?
read -r
answer=$REPLY

re='[0-9]+'

if [[ $answer =~ $re ]];  # need gap between $val, or it will consider as together!
	then
		echo You\'ve enter a number inside string, which is not a name, bye bye!
	else
		echo Hello $answer!
fi
