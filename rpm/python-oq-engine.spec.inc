%define oqrepo ##_repo_##
%define oqversion ##_version_##
%define oqrelease ##_release_##
%define oqname python-%{oqrepo}
%define oqtimestamp ##_timestamp_##

Summary: Computes hazard, risk and socio-economic impact of earthquakes
Name: %{oqname}
Version: %{oqversion}
Release: %{oqtimestamp}_%{oqrelease}
Source0: %{oqrepo}-%{oqversion}-%{oqrelease}.tar.gz
License: AGPLv3
Group: Applications/Engineering
BuildRoot: %{_tmppath}/%{oqname}-%{oqversion}-%{oqrelease}-buildroot
Prefix: %{_prefix}
BuildArch: noarch
Vendor: The GEM OpenQuake team <devops@openquake.org>
Url: http://github.com/gem/oq-engine

%define common_deps python numpy scipy python-shapely python-psutil python-mock python-futures h5py python-amqp python-celery python-psycopg2 python-django rabbitmq-server postgresql-server postgis sudo
%define oqlib_deps python-oq-hazardlib >= 0.15.0 python-oq-risklib >= 0.8.0
%define dev_deps python-nose python-coverage
Requires: %{common_deps}
Requires: %{oqlib_deps}
Requires: %{dev_deps}

# Since tests are currently not enabled during the packaging phase, the following build dependencies are not required
#BuildRequires: %{common_deps}
#BuildRequires: %{dev_deps}
#BuildRequires: %{oqlib_deps}

BuildRequires: python-setuptools

%description

OpenQuake is an open source application that allows users to
compute seismic hazard and seismic risk of earthquakes on a global scale.

Please note: the /usr/bin/oq-engine script requires a celeryconfig.py file in
the PYTHONPATH; when using binary packages, if a celeryconfig.py is not
available the OpenQuake Engine default celeryconfig.py, located in
/usr/share/openquake/engine, is used.

Copyright (C) 2010-2015, GEM Foundation.


%prep
%setup -n %{oqrepo}-%{oqversion}-%{oqrelease} -n %{oqrepo}-%{oqversion}-%{oqrelease}

%build
python setup.py build

%check
#nosetests -v -a '!slow' --with-doctest --with-coverage --cover-package=openquake.engine

%install
python setup.py install --single-version-externally-managed -O1 --root=%{buildroot} --record=INSTALLED_FILES
# create directories where the files will be located
mkdir -p %{buildroot}/%{_sysconfdir}/openquake
mkdir -p %{buildroot}/%{_datadir}/openquake/engine
install -m 644 celeryconfig.py %{buildroot}/%{_datadir}/openquake/engine
install -m 644 openquake.cfg %{buildroot}/%{_sysconfdir}/openquake

%clean
rm -rf %{buildroot}

%files -f INSTALLED_FILES
%defattr(-,root,root)
%doc README.md LICENSE CONTRIBUTORS.txt doc
%{_datadir}/openquake/engine/celeryconfig.py*
%{_sysconfdir}/openquake/openquake.cfg

%changelog
* %(date -d @%{oqtimestamp} '+%a %b %d %Y') GEM Automatic Packager <gem-autopack@openquake.org> %{oqversion}-%{oqtimestamp}_%{oqrelease}
– Unstable release of %{oqname}