#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

import pygame
from pygame.sprite import Group

from settings import Settings
from game_stats import GameStatus
from scoreboard import Scoreboard
from button import Button
from ship import Ship
import game_funcions as gf



def run_game():
    # 初始化pygame游戏、设置和屏幕对象
    pygame.init()
    ai_settings=Settings()
    screen=pygame.display.set_mode((ai_settings.screen_width,ai_settings.screen_height))
    pygame.display.set_caption(ai_settings.name)

    #创建Play按钮
    play_button=Button(ai_settings,screen,"Play")

    #创建一个用于存储游戏统计信息的实例
    stats=GameStatus(ai_settings)
    #创建记分牌
    sb=Scoreboard(ai_settings,screen,stats)

    #创建一艘飞船
    ship=Ship(ai_settings,screen)
    #创建子弹编组
    bullets=Group()
    #创建外星人编组
    aliens=Group()

    #创建外星人群
    gf.create_fleet(ai_settings,screen,ship,aliens)

    # 开始游戏的主循环
    while True:
        #监视键盘和鼠标事件
        gf.check_events(ai_settings,screen,stats,sb,play_button,
                        ship,aliens,bullets)

        if stats.game_active:
            ship.update()
            gf.update_bullets(ai_settings, screen, stats, sb,
                              ship, aliens, bullets)
            gf.update_aliens(ai_settings, screen, stats, sb,
                             ship, aliens, bullets)

        #print("The number of bullets is :%d " %len(bullets))
        #每次循环时都重绘屏幕与物体
        gf.update_screen(ai_settings,screen,stats,sb,ship,
                         aliens,bullets,play_button)

        #添加等待时间
        time.sleep(0.03)

run_game()

