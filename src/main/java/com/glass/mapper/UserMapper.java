package com.glass.mapper;


import com.glass.pojo.User;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface UserMapper {

    public int insert(@Param("username")String username, @Param("password")String password, @Param("deviceId")String deviceId);
    List<User> selectToLogin(@Param("username")String username,@Param("password")String password);

    @Select("SELECT * from user_info WHERE  username=#{username}")
    List<User> selectByUsername(String username);
}
