#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from time import sleep

import pygame

from bullet import Bullet
from alien import Alien

def check_keydown_events(event,ai_settings,screen,ship,bullets):
    """键盘按下"""
    if event.key == pygame.K_RIGHT:
        # 飞船向右移动
        ship.moving_right = True
    elif event.key == pygame.K_LEFT:
        # 飞船向左移动
        ship.moving_left = True
    elif event.key==pygame.K_SPACE:
        #创建一颗新子弹,并将其加入到编组bullets中
        fire_bullet(ai_settings,screen,ship,bullets)
    elif event.key==pygame.K_ESCAPE or event.key==pygame.K_q:
        sys.exit()



def check_keyup_events(event,ship):
    """键盘松开"""
    if event.key == pygame.K_RIGHT:
        ship.moving_right = False
    elif event.key == pygame.K_LEFT:
        ship.moving_left = False


def check_events(ai_settings,screen,stats,sb,play_button,ship,alients,bullets):
    """响应键盘和鼠标事件"""
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()
        elif event.type==pygame.KEYDOWN:
            check_keydown_events(event,ai_settings,screen,ship,bullets)
        elif event.type==pygame.KEYUP:
            check_keyup_events(event,ship)
        elif event.type==pygame.MOUSEBUTTONDOWN:
            mouse_x,mouse_y=pygame.mouse.get_pos()
            check_play_button(ai_settings,screen,stats,sb,play_button,ship,
                              alients,bullets,mouse_x,mouse_y)



def check_play_button(ai_settings,screen,stats,sb,play_button,
                      ship,aliens,bullets,mouse_x,mouse_y):
    """在玩家单击Play按钮时开始新游戏"""
    button_clicked=play_button.rect.collidepoint(mouse_x,mouse_y)
    if button_clicked and not stats.game_active:
        #重置游戏设置
        ai_settings.initialize_dyanmic_settings()
        #隐藏光标
        pygame.mouse.set_visible(False)
        #重置游戏统计信息
        stats.reset_stats()
        stats.game_active=True
        #重置记分牌图像
        sb.prep_score()
        sb.prep_high_score()
        sb.prep_level()
        sb.prep_ships()

        #清空外星人列表和子弹列表
        aliens.empty()
        bullets.empty()
        #创建一群新的外星人,并让飞船居中
        create_fleet(ai_settings,screen,ship,aliens)
        ship.center_ship()




def update_screen(ai_settings, screen, stats, sb, ship,
                  aliens, bullets, play_button):
    """更新屏幕上的图像，并切换到新屏幕"""
    #每次循环都重绘屏幕
    screen.fill(ai_settings.bg_color)
    # 在飞船和外星人后面重绘所有子弹
    for bullet in bullets.sprites():
        bullet.draw_buller()
    # 重绘飞船
    ship.blitme()
    #重绘外星人
    aliens.draw(screen)
    #显示得分
    sb.show_score()

    #如果游戏处于非活动状态,就显示Play按钮
    if not stats.game_active:
        play_button.draw_button()

    #让最近绘制的屏幕可见
    pygame.display.flip()


def update_bullets(ai_settins,screen,stats,sb,
                   ship,aliens,bullets):
    """更新子弹的位置"""
    bullets.update()
    # 删除已消失的子弹
    for bullet in bullets.copy():
        if bullet.rect.bottom <= 0:
            bullets.remove(bullet)

    check_bullet_alien_collisions(ai_settins,screen,stats,sb,ship,aliens,bullets)


def fire_bullet(ai_settings, screen, ship, bullets):
    """若还没有达到限制,就发射一颗新子弹"""
    #创建新子弹,并将其加入到编组bullets中
    if len(bullets) < ai_settings.bullet_allowed:
        new_bullet = Bullet(ai_settings, screen, ship)
        bullets.add(new_bullet)

def get_number_aliens_x(ai_settings,alien_width):
    """计算每行可容纳多少个外星人"""
    # 屏幕左右两边分别留下一定宽度(外星人宽度)
    available_space_x=ai_settings.screen_width-2*alien_width
    # 外星人之间的间距为外星人的宽度
    number_aliens_x=int(available_space_x/(2*alien_width))
    return  number_aliens_x

def get_number_rows(ai_settings, ship_height, alien_height):
    """计算屏幕可容奶多少行外星人"""
    available_space_y=(ai_settings.screen_height-3*alien_height-ship_height)
    number_rows=int(available_space_y/(2*alien_height))
    return number_rows

def create_alien(ai_settings, screen, aliens, alien_number, row_number):
    """创建一个外星人并将其放在当前行"""
    alien = Alien(ai_settings, screen)
    alien_width=alien.rect.width
    alien.x = alien_width + 2 * alien_width * alien_number
    alien.rect.x = alien.x
    alien.rect.y=alien.rect.height+2*alien.rect.height*row_number
    aliens.add(alien)


def create_fleet(ai_settings, screen, ship, aliens):
    """创建外星人群"""
    #创建一个外星人,并计算一行可容纳多少个外星人
    alien=Alien(ai_settings,screen)
    number_aliens_x=get_number_aliens_x(ai_settings,alien.rect.width)
    number_rows=get_number_rows(ai_settings,ship.rect.height,alien.rect.height)

    #创建外星人群
    for row_number in range(number_rows):
        for alien_number in range(number_aliens_x):
            create_alien(ai_settings, screen, aliens, alien_number,row_number)


def check_fleet_edges(ai_settings, aliens):
    """有外星人到达边缘时采取相应的措施
    向下移动并改变移动方向
    """
    for alien in aliens.sprites():
        if alien.check_edge():
            change_fleet_direction(ai_settings,aliens)
            break



def change_fleet_direction(ai_settings,aliens):
    """"将整群外星人下移,并改变它们的方向"""
    for alien in aliens.sprites():
        alien.rect.y+=ai_settings.fleet_drop_speed
    ai_settings.fleet_direction*=-1


def ship_hit(ai_settings,screen,stats,sb,ship,aliens,bullets):
    """响应飞船被外星人撞到的情况"""
    if stats.ships_left>0:
        #将ships_left减去1
        stats.ships_left-=1
        #更新剩余飞船数目
        sb.prep_ships()

        #清空外星人列表和子弹列表
        aliens.empty()
        bullets.empty()

        #创建一群新的外星人,并将飞船放到屏幕底部中间
        create_fleet(ai_settings,screen,ship,aliens)
        ship.center_ship()

        #暂停
        sleep(0.5)
    else:
        stats.game_active=False
        #使光标可见
        pygame.mouse.set_visible(True)


def check_aliens_bottom(ai_settings,screen,stats,sb,ship,aliens,bullets):
    """检查是否有外星人到达了屏幕底部"""
    screen_rect=screen.get_rect()
    for alien in aliens.sprites():
        if alien.rect.bottom>=screen_rect.bottom:
            #像飞船被撞倒一样进行处理
            ship_hit(ai_settings,screen,stats,sb,ship,aliens,bullets)
            break


def update_aliens(ai_settings,screen,stats,sb,ship,aliens,bullets):
    """检查是否有外星人位于屏幕边缘,并更新外星人群中所有外星人的位置"""
    check_fleet_edges(ai_settings,aliens)
    aliens.update()

    #检测外星人和飞船碰撞
    if pygame.sprite.spritecollideany(ship,aliens):
        ship_hit(ai_settings,screen,stats,sb,ship,aliens,bullets)

    #检查是否有外星人达到屏幕底部
    check_aliens_bottom(ai_settings,screen,stats,sb,ship,aliens,bullets)


def check_bullet_alien_collisions(ai_settings, screen, stats, sb,
                                  ship, aliens, bullets):
    """响应子弹和外星人碰撞"""
    # 检查是否有子弹击中了外星人
    # 如果击中, 就删除相应的子弹和外星人
    # sprite.groupcollide()返回一个字典,其中包含发生了碰撞的子弹和外星人,在这个字典中,每个键都是一颗子弹,而相应的值都是被击中的外星人
    collisions=pygame.sprite.groupcollide(bullets,aliens,True,True)

    """
    如果字典collisions存在, 我们就遍历其中的所有值,
    每个值都是一个列表, 包含被同一颗子弹击中的所有外星人
    """
    if collisions:
        for aliens in collisions.values():
            stats.score+=ai_settings.alien_points*len(aliens)
            sb.prep_score()
        check_high_score(stats,sb)

    if len(aliens)==0:
        #删除现有的所有的子弹,并创建一个新的外星人群
        bullets.empty()
        ai_settings.increase_speed()
        create_fleet(ai_settings,screen,ship,aliens)
        #提高等级
        stats.level+=1
        sb.prep_level()


def check_high_score(stats,sb):
    """检查是否诞生了新的最高得分"""
    if stats.score > stats.high_score:
        stats.high_score=stats.score
        sb.prep_high_score()




