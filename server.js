let app = require('express')();
let http = require('http').Server(app);
let https = require('https');
let io = require('socket.io')(http);

var roles = {};

io.on('connection', function(socket){
	socket.on('disconnect', function(){
		delete roles[socket.username];

		io.emit('users-changed', {
			user: socket.username, //nickname
			event: 'left'
		});
	});

	socket.on('set-username', function(username, role){
		roles[username] = role;

		socket.username = username;
		io.emit('users-changed', {
			user: username, //nickname
			event: 'joined'
		});
	});

	socket.on('add-message', function(message){
		io.emit('message', {
			text: message.text,
			from: message.from,
			role: roles[message.from],
			created: (new Date())
		});
	});
});

http.listen(3001, function(){
	console.log('listening');
});