#include <string>

#include <ros/ros.h>
#include <gazebo_msgs/ContactsState.h>
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>

namespace gazebo
{
/// \brief An example plugin for a contact sensor.
class ContactPlugin : public SensorPlugin
{
public:
  /// \brief Constructor.
  ContactPlugin() : SensorPlugin()
  {
    topicName = "contacts";
  }

  /// \brief Destructor.
  virtual ~ContactPlugin()
  {}

  /// \brief Load the sensor plugin.
  /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
  /// \param[in] _sdf SDF element that describes the plugin.
  virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
  {
    // Make sure the ROS node for Gazebo has already been initialized
    if (!ros::isInitialized()) {
      ROS_INFO("ROS should be initialized first!");
      return;
    }

    if (!_sensor) {
      gzerr << "Invalid sensor pointer.\n";
    }

    // Get the parent sensor.
    this->parentSensor =
      std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);

    // Make sure the parent sensor is valid.
    if (!this->parentSensor) {
      gzerr << "ContactPlugin requires a ContactSensor.\n";
      return;
    }

    ROS_INFO("The Sonar plugin has been loaded!");

    node_handle_ = new ros::NodeHandle("");
    pub_ = node_handle_->advertise<gazebo_msgs::ContactsState>(topicName, 1);

    // Connect to the sensor update event.
    this->updateConnection =
      this->parentSensor->ConnectUpdated(std::bind(&ContactPlugin::OnUpdate, this));

    // Make sure the parent sensor is active.
    this->parentSensor->SetActive(true);
  }

private:
  // FIXME: copy all the members, not just collision-link names
  gazebo_msgs::ContactsState gzContacts2rosContacts(msgs::Contacts contacts)
  {
    gazebo_msgs::ContactsState rContacts;
    gazebo_msgs::ContactState contactState;

    for (unsigned int i = 0; i < contacts.contact_size(); ++i) {
      contactState.collision1_name = contacts.contact(i).collision1();
      contactState.collision2_name = contacts.contact(i).collision2();

      rContacts.states.push_back(contactState);
    }

    return rContacts;
  }

  /// \brief Callback that receives the contact sensor's update signal.
  virtual void OnUpdate()
  {
    // Get all the contacts.
    msgs::Contacts contacts;
    contacts = this->parentSensor->Contacts();

    if (contacts.contact_size() > 0) {
      pub_.publish(gzContacts2rosContacts(contacts));
    }
  }

  /// \brief Pointer to the contact sensor
  sensors::ContactSensorPtr parentSensor;

  /// \brief Connection that maintains a link between the contact sensor's
  /// updated signal and the OnUpdate callback.
  event::ConnectionPtr updateConnection;

  ros::NodeHandle *node_handle_;
  ros::Publisher pub_;
  std::string topicName;
};

GZ_REGISTER_SENSOR_PLUGIN(ContactPlugin)
}
