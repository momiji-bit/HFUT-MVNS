package com.glass.service;

import com.alibaba.fastjson.JSONObject;
import com.glass.mapper.DeviceMapper;
import com.glass.mapper.UserMapper;
import com.glass.pojo.Device;
import com.glass.util.SqlSessionFactoryUtils;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;

import java.util.List;

public class DeviceService {
    SqlSessionFactory sqlSessionFactory= SqlSessionFactoryUtils.getSqlSessionFactory();


    public List<String> getAllDeviceId(){
        SqlSession sqlSession= sqlSessionFactory.openSession();
        DeviceMapper deviceMapper = sqlSession.getMapper(DeviceMapper.class);
        List<String> deviceids = deviceMapper.selectAllDeviceid();

        sqlSession.close();
        return deviceids;
    }

    public Device getDeviceByDeviceid(String deviceId){
        SqlSession sqlSession= sqlSessionFactory.openSession();
        DeviceMapper deviceMapper = sqlSession.getMapper(DeviceMapper.class);
        Integer id = deviceMapper.selectIdByDeviceid(deviceId);
        Device device = deviceMapper.selectById(id);
        sqlSession.close();
        return device;
    }
    public Integer getIdByDeviceId(String deviceId){

        SqlSession sqlSession= sqlSessionFactory.openSession();
        DeviceMapper deviceMapper = sqlSession.getMapper(DeviceMapper.class);
        sqlSession.close();
        Integer count = deviceMapper.selectIdByDeviceid(deviceId);
        sqlSession.close();
        return count;
    }

    public boolean updateStatusByDeviceId(String deviceId, String collide,String edge,String gesture,String broadcast ,String navigation ){
        SqlSession sqlSession= sqlSessionFactory.openSession();
        DeviceMapper deviceMapper = sqlSession.getMapper(DeviceMapper.class);

        Integer count = deviceMapper.updateStasusByDeviceId(deviceId, collide, edge, gesture, broadcast, navigation);
        sqlSession.commit();
        sqlSession.close();
        return count>0 ;
    }
}
