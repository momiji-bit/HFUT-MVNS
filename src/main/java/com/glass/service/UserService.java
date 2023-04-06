package com.glass.service;

import com.glass.mapper.DeviceMapper;
import com.glass.mapper.UserMapper;
import com.glass.pojo.Device;
import com.glass.pojo.User;
import com.glass.util.SqlSessionFactoryUtils;
import org.apache.ibatis.session.SqlSessionFactory;
import com.alibaba.fastjson.JSONObject;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UserService {
    SqlSessionFactory sqlSessionFactory= SqlSessionFactoryUtils.getSqlSessionFactory();
    UserMapper userMapper=sqlSessionFactory.openSession(true).getMapper(UserMapper.class);
    DeviceMapper deviceMapper = sqlSessionFactory.openSession().getMapper(DeviceMapper.class);

    /**
     * 注册
     * @param username
     * @param password
     * @param deviceId
     * @return
     */
    public String register(String username,String password,String deviceId){
        List<User> users = userMapper.selectByUsername(username);
        JSONObject jsonObject = new JSONObject();
        if(users.size()>0){

            System.out.println(users.get(0));
            jsonObject.put("success",false);
            jsonObject.put("username_msg","用户名已被使用");
        }   else {
            Device device = deviceMapper.findDeviceByDeviceid(deviceId);
            System.out.println("decice:  "+device);
            if(device ==null){
                //没有该设备
                jsonObject.put("success",false);
                jsonObject.put("deviceid_msg","没有该设备，请检查是否输入错误");
            }   else {
                //用户名和设备id合法
                int count = userMapper.insert(username, password, deviceId);
                System.out.println(count);
                if(count>0){//注册成功
                    jsonObject.put("success",true);
                    jsonObject.put("msg","注册成功");
                }else{
                    jsonObject.put("success",false);
                    jsonObject.put("msg","注册失败");
                }
            }
        }
        return jsonObject.toJSONString();
    }
}
