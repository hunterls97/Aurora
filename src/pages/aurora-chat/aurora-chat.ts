import { Component } from '@angular/core';
import { IonicPage, NavController, NavParams, ToastController  } from 'ionic-angular';
import { Socket } from 'ng-socket-io';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs/Observable';

/**
 * Generated class for the AuroraChatPage page.
 *
 * See https://ionicframework.com/docs/components/#navigation for more info on
 * Ionic pages and navigation.
 */

@IonicPage()
@Component({
  selector: 'page-aurora-chat',
  templateUrl: 'aurora-chat.html',
})
export class AuroraChatPage {
	messages = [];
	username = '';
	message = '';
	roles = {};
	isManager = null;

	constructor(public navCtrl: NavController, public navParams: NavParams, private socket: Socket, private toastCtrl: ToastController, public http: HttpClient) {
		this.username = this.navParams.get('username');
		this.isManager = this.navParams.get('isManager');

		this.getMessages().subscribe(message => {
			this.messages.push(message);
		});

		this.getUsers().subscribe(data => {
			let user = data['user'];

			if (data['event'] === 'left') {
				this.showToast('User left: ' + user);
			} 
			else {
				this.showToast('User joined: ' + user);
			}
		});
	}

	sendMessage() {
		let body = {
			query: this.message
		}

		this.socket.emit('add-message', {text: this.message, from: this.username});

		let headers = new HttpHeaders();
		headers.append("Content-Type", "application/json");
		//this.http.setHeader("Content-Type", "application/json");
		if(this.isManager == 0 || 1){
			this.http.post('http://127.0.0.1:8000/aurora/index/', JSON.stringify(body), {
				headers: headers,
				responseType: 'text',
				withCredentials: false
			})
				.subscribe(res => {
					this.socket.emit('add-message', {text: res, from: 'Aurora'});
				}, (err) => {
					console.log(err);
					this.socket.emit('add-message', {text: 'error: ' + err, from: 'Aurora'});
				});
		}

		this.message = '';
	}

	getMessages() {
		let observable = new Observable(observer => {
			this.socket.on('message', (data) => {
				observer.next(data);
			});
		});

		return observable;
	}

	getUsers() {
		let observable = new Observable(observer => {
			this.socket.on('users-changed', (data) => {
				observer.next(data);
			});
		});

		return observable;
	}

	ionViewWillLeave() {
		this.socket.disconnect();
	}

	showToast(msg) {
		let toast = this.toastCtrl.create({
			message: msg,
			duration: 2000
		});

		toast.present();
	}

	ionViewDidLoad() {
		console.log('ionViewDidLoad AuroraChatPage');
	}

}
