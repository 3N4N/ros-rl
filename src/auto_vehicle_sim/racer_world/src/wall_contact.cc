#include <string>

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
  {}

  /// \brief Destructor.
  virtual ~ContactPlugin()
  {}

  /// \brief Load the sensor plugin.
  /// \param[in] _sensor Pointer to the sensor that loaded this plugin.
  /// \param[in] _sdf SDF element that describes the plugin.
  virtual void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
  {
    // Get the parent sensor.
    this->parentSensor =
      std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);

    // Make sure the parent sensor is valid.
    if (!this->parentSensor) {
      gzerr << "ContactPlugin requires a ContactSensor.\n";
      return;
    }

    // Connect to the sensor update event.
    this->updateConnection =
      this->parentSensor->ConnectUpdated(std::bind(&ContactPlugin::OnUpdate, this));

    // Make sure the parent sensor is active.
    this->parentSensor->SetActive(true);
  }

private:
  /// \brief Callback that receives the contact sensor's update signal.
  virtual void OnUpdate()
  {
    // Get all the contacts.
    msgs::Contacts contacts;
    contacts = this->parentSensor->Contacts();
    for (unsigned int i = 0; i < contacts.contact_size(); ++i)
    {
      std::cout << "Collision between[" << contacts.contact(i).collision1()
                << "] and [" << contacts.contact(i).collision2() << "]\n";

      for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
      {
        std::cout << j << "  Position:"
                  << contacts.contact(i).position(j).x() << " "
                  << contacts.contact(i).position(j).y() << " "
                  << contacts.contact(i).position(j).z() << "\n";
        std::cout << "   Normal:"
                  << contacts.contact(i).normal(j).x() << " "
                  << contacts.contact(i).normal(j).y() << " "
                  << contacts.contact(i).normal(j).z() << "\n";
        std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";
      }
    }
  }

  /// \brief Pointer to the contact sensor
  sensors::ContactSensorPtr parentSensor;

  /// \brief Connection that maintains a link between the contact sensor's
  /// updated signal and the OnUpdate callback.
  event::ConnectionPtr updateConnection;
};

GZ_REGISTER_SENSOR_PLUGIN(ContactPlugin)
}

