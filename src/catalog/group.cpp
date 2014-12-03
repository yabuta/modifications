/* This file is part of VoltDB.
 * Copyright (C) 2008-2014 VoltDB Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with VoltDB.  If not, see <http://www.gnu.org/licenses/>.
 */

/* WARNING: THIS FILE IS AUTO-GENERATED
            DO NOT MODIFY THIS SOURCE
            ALL CHANGES MUST BE MADE IN THE CATALOG GENERATOR */

#include <cassert>
#include "group.h"
#include "catalog.h"
#include "userref.h"

using namespace catalog;
using namespace std;

Group::Group(Catalog *catalog, CatalogType *parent, const string &path, const string &name)
: CatalogType(catalog, parent, path, name),
  m_users(catalog, this, path + "/" + "users")
{
    CatalogValue value;
    m_childCollections["users"] = &m_users;
    m_fields["admin"] = value;
    m_fields["defaultproc"] = value;
    m_fields["defaultprocread"] = value;
    m_fields["sql"] = value;
    m_fields["sqlread"] = value;
    m_fields["allproc"] = value;
}

Group::~Group() {
    std::map<std::string, UserRef*>::const_iterator userref_iter = m_users.begin();
    while (userref_iter != m_users.end()) {
        delete userref_iter->second;
        userref_iter++;
    }
    m_users.clear();

}

void Group::update() {
    m_admin = m_fields["admin"].intValue;
    m_defaultproc = m_fields["defaultproc"].intValue;
    m_defaultprocread = m_fields["defaultprocread"].intValue;
    m_sql = m_fields["sql"].intValue;
    m_sqlread = m_fields["sqlread"].intValue;
    m_allproc = m_fields["allproc"].intValue;
}

CatalogType * Group::addChild(const std::string &collectionName, const std::string &childName) {
    if (collectionName.compare("users") == 0) {
        CatalogType *exists = m_users.get(childName);
        if (exists)
            return NULL;
        return m_users.add(childName);
    }
    return NULL;
}

CatalogType * Group::getChild(const std::string &collectionName, const std::string &childName) const {
    if (collectionName.compare("users") == 0)
        return m_users.get(childName);
    return NULL;
}

bool Group::removeChild(const std::string &collectionName, const std::string &childName) {
    assert (m_childCollections.find(collectionName) != m_childCollections.end());
    if (collectionName.compare("users") == 0) {
        return m_users.remove(childName);
    }
    return false;
}

const CatalogMap<UserRef> & Group::users() const {
    return m_users;
}

bool Group::admin() const {
    return m_admin;
}

bool Group::defaultproc() const {
    return m_defaultproc;
}

bool Group::defaultprocread() const {
    return m_defaultprocread;
}

bool Group::sql() const {
    return m_sql;
}

bool Group::sqlread() const {
    return m_sqlread;
}

bool Group::allproc() const {
    return m_allproc;
}

