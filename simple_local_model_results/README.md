Dillon's comments

I realized you guys had to rush the last minute so I don't expect you to have
implemented everything yet. If there's some feedback I provide that you would
have done had you had more time then don't worry about it!

- The graphs look good!

`image_batch_generator` Looks like you handle the last batch problem. That is
the most common error I have when writing loading functions.

`parameter_dict` Parameter dictionaries are good. Another common way to pass
parameters is through command line arguments which makes it easier to run code
with shell scripts. Tensorflow and PyTorch both have arg parsers that turn
command line arguments into dictionary-like objects that you can pass around. I
run a lot of scripts with nohup and sometimes run multiple processes in a shell
script.

- Overall this code looks really good!
