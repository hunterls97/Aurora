import { Component } from '@angular/core';
import { NavController } from 'ionic-angular';
import { Socket } from 'ng-socket-io';

@Component({
  selector: 'page-home',
  templateUrl: 'home.html'
})
export class HomePage {
	username = '';

	constructor(public navCtrl: NavController, private socket: Socket) {

	}

	joinChat(isManager){
		this.socket.connect();
		this.socket.emit('set-username', this.username, isManager);
		this.navCtrl.push('AuroraChatPage', {username: this.username, isManager: isManager});
	}

}
