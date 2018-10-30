import { NgModule } from '@angular/core';
import { IonicPageModule } from 'ionic-angular';
import { AuroraChatPage } from './aurora-chat';

@NgModule({
  declarations: [
    AuroraChatPage,
  ],
  imports: [
    IonicPageModule.forChild(AuroraChatPage),
  ],
})
export class AuroraChatPageModule {}
