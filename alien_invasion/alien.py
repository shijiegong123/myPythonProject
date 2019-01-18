#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pygame
from pygame.sprite import Sprite
import random

alien_image_list=['images/alien_72px.png']
alien_image_list.append('images/Lego_Alien_72px.png')
alien_image_list.append('images/Lego_Alien_Lord_72px.png')

class Alien(Sprite):
    """定义外星人类"""
    def __init__(self,ai_setting,screen):
        """初始化外星人并设置其起始位置"""
        super(Alien,self).__init__()
        self.screen=screen
        self.ai_setting=ai_setting

        #加载外星人图像,并设置其rect属性
        self.image=pygame.image.load(alien_image_list[random.randint(0,2)])
        self.rect=self.image.get_rect()

        #每个外星人最初都在屏幕左上角附近
        self.rect.x=self.rect.width
        self.rect.y=self.rect.height

        #存储外星人的准确位置
        self.x=float(self.rect.x)

    def blitme(self):
        """在指定的位置绘制外星人"""
        self.screen.blit(self.image,self.rect)

    def check_edge(self):
        """如果外星人位于屏幕边缘,就返回True"""
        screen_rect=self.screen.get_rect()
        if self.rect.right >= screen_rect.right:
            return True
        elif self.rect.left<=0:
            return True


    def update(self):
        """向左或向右移动外星人"""
        self.x+=self.ai_setting.alien_speed_factor*self.ai_setting.fleet_direction
        self.rect.x=self.x







