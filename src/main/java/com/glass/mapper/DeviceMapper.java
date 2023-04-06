package com.glass.mapper;

import com.glass.pojo.Device;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

public interface DeviceMapper {
    @Select("SELECT * from device_status WHERE device_id = #{deviceId}")
    public Device findDeviceByDeviceid(String deviceId);

    @Select("select device_id from device_status;")
    List<String> selectAllDeviceid();

    @Select("select id from device_status where device_id=#{deviceId}")
    Integer selectIdByDeviceid(String deviceId);

    @Select("SELECT * from device_status where id=#{id}")
    Device selectById(int id);

    @Update(  "UPDATE device_status SET collide = #{collide}, edge=#{edge} , gesture=#{gesture}, broadcast=#{broadcast}, navigation=#{navigation} WHERE device_id= #{deviceId}")
    Integer updateStasusByDeviceId(@Param("deviceId") String deviceId,@Param("collide") String collide,@Param("edge") String edge ,@Param("gesture") String gesture, @Param("broadcast") String broadcast,@Param("navigation") String navigation);
}
